import torch
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.helper_functions.helper_functions import (
    mAP,
    CocoDetection,
    CutoutPIL,
    ModelEma,
    add_weight_decay,
)
import argparse
from PIL import Image
import numpy as np
import pickle
from src.models import create_model
from src.helper_functions.helper_functions import AverageMeter, CocoDetection
from voc import *
from nus import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import random
from torch.optim import lr_scheduler
import torch.nn.functional as F
import os
from tqdm import tqdm
import multiprocessing
import torchvision.models as models
import torch.nn as nn
import timeit
from mpi4py import MPI
import time
import torchvision.models as models
import nuswide

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser(description="ASL MS-COCO Inference on a single image")

parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--m", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--dataset_type", type=str, default="NUS-WIDE")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weightdecay", type=float, default=0)
parser.add_argument("--step", type=int, default=5)  # 40
parser.add_argument("--convepoch", type=int, default=30)  # 40
parser.add_argument("--epoch", type=int, default=10)  # 100
parser.add_argument("--gamma", type=float, default=0.5)  # 0.1
parser.add_argument("--sigma", type=float, default=1)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--load_w", type=int, default=0)
parser.add_argument('--certification', default = "bagging", type=str, help='certification')
parser.add_argument("--model_name", type=str, default="tresnet_m")
parser.add_argument("--pretrained", type=str, default="./moco_v2_800ep_pretrain.pth.tar")
parser.add_argument("--log", type=str, default="train_voc.txt")

parser.add_argument('--num_partitions', default = 80, type=int, help='number of partitions')
parser.add_argument('--start_partition', required=False, type=int, help='partition number')
parser.add_argument('--num_partition_range', default=10, type=int, help='number of partitions to train')

args = parse_args(parser)
from torch.utils.data import Dataset
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        target = self.targets[idx]
        return feature, target


def focal_binary_cross_entropy(logits, targets, gamma=2):
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = loss.mean()
    return loss
def eval(model, data_loader, sigma,device):

    model.eval()
    precision = 0
    recall = 0
    precision = 0
    count = 0
    for i, (input, target) in enumerate(data_loader):
        
        # target 1,-1
        if len(target.shape) == 3:
            targ = target.max(dim=1)[0].to(device)
        else:
            targ = F.relu(target).to(device)
        #targ[:,0] =0
        with torch.no_grad():
            # hard classifier: => >0 ,scorer sigmoid
            output = model(input.to(device))
            #print(torch.argmax(output,dim = 1))
            #print(targ)
            #print(torch.argmax(output, dim=1))
            topk_indices = torch.topk(output, k=3, dim=1)[1]
            #print(topk_indices)
            # create binary mask based on top-k indices
            mask = torch.zeros_like(output)
            mask.scatter_(1, topk_indices, 1)
            #print(mask)
            #print(targ)
            #rint(np.sum(output*targ))
            #print(targ,output)
            #print(np.sum(output*targ))
            if targ.sum()!=0:
                precision +=(((targ* mask).sum().item())/3)
                recall+= ((targ* mask).sum().item()/targ.sum().item())
                count+=1
            #print(torch.eq(targ, mask).sum().item(), torch.eq(targ, mask).sum().item())
            #print(precision,recall)
    #print(precision,recall)
    print("=====DEVICE: ", device, "Precision:",precision/count,flush=True)
    print("=====DEVICE: ", device, "Recall:",recall/count,flush=True)
    return 1

def count(data_loader, device):

    targs = []
    for i, (input, target) in enumerate(data_loader):

        # target 1,-1
        if len(target.shape) == 3:
            targ = target.max(dim=1)[0].to(device)
        else:
            targ = F.relu(target).to(device)
        targs.append(targ)
    #print(precision,recall)
    targs = torch.cat(targs, dim=0)
    print(targs.shape)
    print(100/targs.sum(dim = 0))
    return 100/targs.sum(dim = 0)

def extract_feature(model, train_loader, device):
    embeddings_list = []
    target_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            print(i,flush = True)
            output = model(input.to(device))
            #print(output.shape,flush = True)
            embeddings_list.append(output.squeeze().to("cpu"))
            if len(target.shape) == 3:
                target = target.max(dim=1)[0].to(device)
            else:
                target = F.relu(target).to(device)
            print(target.shape)
            target_list.append(target.to("cpu"))
            print(embeddings_list[0].shape)
    embeddings_list = torch.cat(embeddings_list, dim=0).contiguous()
    target_list = torch.cat(target_list, dim=0).contiguous()
    print(embeddings_list.shape, target_list.shape)
    return torch.utils.data.TensorDataset(embeddings_list.detach(),target_list.detach())
def train(model, train_loader, optimizer, scheduler, weights,device):
    #print("model is on:",next(model.parameters()).get_device())
    start = timeit.default_timer()

    loss_func = AsymmetricLossOptimized()
    #loss_func = torch.nn.BCEWithLogitsLoss()
    #loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
    #AsymmetricLossOptimized()
    model.train()
    tot = 0
    for i, (input, target) in enumerate(train_loader):
        #print(device,str(i))
        optimizer.zero_grad()
        # target 1,-1
        if len(target.shape) == 3:
            target = target.max(dim=1)[0].to(device)
        else:
            target = F.relu(target).to(device)
        #print(target,flush=True)
        #target[:,0] =0
        #print(target,flush=True)
        output = model(input.to(device))
        loss = loss_func(output, target.float())
        loss.backward()
        optimizer.step()
        tot += loss.item()

    #scheduler.step()
    fi = open(args.log, "a")
    #comm.Barrier()


    #print("DEVICE: ", device, " Train Loss tot: ", tot,flush=True)
    #print("DEVICE: ", device,"Train Loss tot: ", tot, file=fi,flush=True)
    
    #Your statements here

    stop = timeit.default_timer()

    print("DEVICE: ", device,'Time: ', stop - start,flush=True)  
    fi.close()

    return 1

def main(index):
    device = torch.device("cuda:"+str(index+5))
    for i in range(torch.cuda.device_count()):
        device_name = f'cuda:{i}'
        print(f'{i} device name:{torch.cuda.get_device_name(torch.device(device_name))}')
    proc_name = multiprocessing.current_process().name
    proc_pid = os.getpid()
    print(f"Worker {proc_name} with PID {proc_pid} is running",flush=True)
    #print(index)
    availble_gpus = list(range(torch.cuda.device_count()))

    print("Validation code for multi-label classification",flush=True)

    torch.cuda.empty_cache()
    # setup model
    print("creating and loading the model...")

    if args.dataset_type == "PASCAL-VOC":  # ML-GCN
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    CutoutPIL(cutout_factor=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        test_transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        args.num_classes = 20
        global_train_dataset = get_voc2007_train(train_transform)
        global_val_dataset = get_voc2007_test(test_transform)
    elif args.dataset_type == "MS-COCO":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    CutoutPIL(cutout_factor=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        test_transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        args.num_classes = 80
        args.do_bottleneck_head = False
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        instances_path_val = os.path.join(
            "./COCO", "annotations/instances_val2014.json"
        )
        val_data_path = os.path.join("./COCO", "val2014")
        global_val_dataset = CocoDetection(
            val_data_path,
            instances_path_val,
            test_transform,
        )
        instances_path_train = os.path.join(
            "./COCO", "annotations/instances_train2014.json"
        )
        train_data_path = os.path.join("./COCO", "train2014")
        global_train_dataset = CocoDetection(
            train_data_path,
            instances_path_train,
            train_transform,
        )
    elif args.dataset_type == "NUS-WIDE":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    CutoutPIL(cutout_factor=0.5),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        test_transform = transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        args.num_classes = 81
        global_train_dataset = nuswide.NUSWIDEClassification('.', 'trainval',transform = train_transform)
        global_val_dataset = nuswide.NUSWIDEClassification('.', 'test',transform = test_transform)
    global_train_loader = torch.utils.data.DataLoader(
            global_train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last = True
        )
    global_val_loader = torch.utils.data.DataLoader(
            global_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last = True
        )


    """
    model = create_model(args)
    print("model created!",index)
    """

    model = models.__dict__['resnet50']()
    num_ftrs = model.fc.in_features

        # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias"]:
            param.requires_grad = False
            # load from pre-trained, before DistributedDataParallel constructor
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.fc"
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            args.start_epoch = 0
            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
                
        # Modify the last layer to have 10 output classes (for example)

    print("model created!")
    encoder = nn.Sequential(*list(model.children())[:-1])
    print(device)
    encoder.to(device)
    global_train_feature_dataset = extract_feature(encoder, global_train_loader, device)
    global_val_feature_dataset = extract_feature(encoder, global_val_loader, device)
    if args.certification == "bagging":
        for k in range(1,501):
            print(device,flush = True)
            print(k,flush = True)
            seed = (500*index+k)*int(time.time()) % 123456789
            print("SEED IS:", seed,flush=True)
            torch.manual_seed(seed)
            idxs=np.arange(len(global_train_feature_dataset))
            idxs = np.random.choice(idxs, size=args.m, replace=True)
            #print("indexes:",idxs)
            train_feature_dataset = torch.utils.data.Subset(global_train_feature_dataset, idxs)

            idxs=np.arange(len(global_val_feature_dataset))
            np.random.shuffle(idxs)
            idxs=idxs[:int(1000)]
            val_feature_dataset = torch.utils.data.Subset(global_val_feature_dataset, idxs)



            #weights = count(count_loader,device)
            fc = nn.Linear(num_ftrs, args.num_classes)

            # init the fc layer
            fc.weight.data.normal_(mean=0.0, std=0.01)
            fc.bias.data.zero_()


            fi = open(args.log, "w")
            fi.close()

            #torch.cuda.set_device(index)

            fc.to(device)

            train_feature_loader = torch.utils.data.DataLoader(
                train_feature_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            val_feature_loader = torch.utils.data.DataLoader(
                val_feature_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            lr = args.lr
            #parameters = add_weight_decay(fc, args.weightdecay)
            # true wd, filter_bias_and_bn
            optimizer = torch.optim.Adam(fc.parameters, lr=lr, weight_decay=0.0001)

            #steps_per_epoch = len(train_loader)
            #scheduler = lr_scheduler.OneCycleLR(
             #   optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=args.epoch
            #)

            for epoch in range(args.epoch):
                t = max(0, args.convepoch - epoch)
                if args.convepoch > 0:
                    sigma = (1 - 1 / (args.convepoch * args.convepoch) * t * t) * args.sigma
                else:
                    sigma = args.sigma
                train(fc, train_feature_loader, optimizer, None, None,device)
                if epoch%10 == 9:
                    eval(fc, val_feature_loader, sigma,device)
                    #eval(model, train_loader, sigma,device)
                    print("DEVICE: ",device, "EPOCH FINISHED: ", epoch,flush=True)
                #eval(model, val_loader, args.sigma,device)
                if epoch == args.epoch-1:
                    if args.dataset_type == "MS-COCO":
                        if not os.path.exists('COCO_moco_models_m='+str(args.m)):
                            os.makedirs('COCO_moco_models_m='+str(args.m))
                        torch.save(fc.state_dict(), "COCO_moco_models_m=%d/save_lr%f_epoch%d_rank%d_seed%d.pth" % (args.m,args.lr, epoch,0,seed))
                    if args.dataset_type == "NUS-WIDE":
                        if not os.path.exists('NUS_moco_models_m='+str(args.m)):
                            os.makedirs('NUS_moco_models_m='+str(args.m))
                        torch.save(fc.state_dict(), "NUS_moco_models_m=%d/save_lr%f_epoch%d_rank%d_seed%d.pth" % (args.m,args.lr, epoch,index,seed))
                    if args.dataset_type == "PASCAL-VOC":
                        if not os.path.exists('VOC_moco_models_m='+str(args.m)):
                            os.makedirs('VOC_moco_models_m='+str(args.m))
                        torch.save(fc.state_dict(), "VOC_moco_models_m=%d/save_lr%f_epoch%d_rank%d_seed%d.pth" % (args.m,args.lr, epoch,index,seed))
    if args.certification =="dpa":

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        dirbase = 'COCO_nin_baseline'
        if (args.zero_seed):
            dirbase += '_zero_seed'

        checkpoint_dir = 'checkpoints'
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        checkpoint_subdir = f'./{checkpoint_dir}/' + dirbase + f'_COCO_moco_partitions_{args.num_partitions}'
        if not os.path.exists(checkpoint_subdir):
            os.makedirs(checkpoint_subdir)
        print("==> Checkpoint directory", checkpoint_subdir)
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        partitions_file = torch.load('partitions_hash_mean_MS-COCO_'+str(args.num_partitions)+'.pth')
        partitions = partitions_file['idx']
        args.start_partition = (rank%2)*(int(args.num_partitions/2))
        args.num_partition_range = int(args.num_partitions/2)

        for part in range(args.start_partition,args.start_partition+args.num_partition_range):
            seed = part
            if (args.zero_seed):
                seed = 0
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            curr_lr = 0.01
            print('\Partition: %d' % part)
            part_indices = torch.tensor(partitions[part])
            print(part_indices.shape)


            fc = nn.Linear(num_ftrs, args.num_classes)

            # init the fc layer
            fc.weight.data.normal_(mean=0.0, std=0.01)
            fc.bias.data.zero_()


            fi = open(args.log, "w")
            fi.close()

            #torch.cuda.set_device(index)

            fc.to(device)

            #print("indexes:",idxs)
            train_feature_dataset = torch.utils.data.Subset(global_train_feature_dataset, part_indices)
            idxs=np.arange(len(global_val_feature_dataset))
            np.random.shuffle(idxs)
            idxs=idxs[:int(1000)]
            val_feature_dataset = torch.utils.data.Subset(global_val_feature_dataset, idxs)
            train_feature_loader = torch.utils.data.DataLoader(
                train_feature_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            val_feature_loader = torch.utils.data.DataLoader(
                val_feature_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            lr = args.lr
            parameters = add_weight_decay(fc, args.weightdecay)
            # true wd, filter_bias_and_bn
            optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0.001)

        # Training
            for epoch in range(args.epoch):
                train(model, trainloader, optimizer, device)
                if epoch%10 == 9:
                    eval(model, val_loader, device)
                    print("DEVICE: ",device, "EPOCH FINISHED: ", epoch)
                #eval(model, val_loader, args.sigma,device)
                if epoch ==9:
                    # Save checkpoint.
                    print('Saving..')
                    state = {
                        'net': model.state_dict(),

                        'partition': part
                    }
                    torch.save(state, checkpoint_subdir + '/partition_'+ str(part)+'.pth')
if __name__ == "__main__":
    """
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    """
    #print(rank)
    main(rank)
    #torch.backends.cudnn.deterministic = True
    """
    processes = []
    for i in range(1):
        process = multiprocessing.Process(target=main,args=(i,))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    """
    """
    ctx = multiprocessing.get_context('spawn')
    
    # Use the context to create a Pool object
    pool = ctx.Pool(processes=4)
    
    # Use the Pool object to run the main function in parallel
    pool.map(main, [0, 1, 2, 3])
    pool.close()
    pool.join()
    """
