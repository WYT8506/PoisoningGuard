from __future__ import print_function
import os
import sys
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
from src.helper_functions.helper_functions import (
    mAP,
    CocoDetection,
    CutoutPIL,
    ModelEma,
    add_weight_decay,
)
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
sys.path.append('./FeatureLearningRotNet/architectures')
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import numpy as np
import random
from mpi4py import MPI
import torchvision.models as models
import timeit
import time
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--num_partitions', default = 80, type=int, help='number of partitions')
parser.add_argument('--start_partition', required=False, type=int, help='partition number')
parser.add_argument('--num_partition_range', default=10, type=int, help='number of partitions to train')
parser.add_argument('--zero_seed', action='store_true', help='Use a random seed of zero (instead of the partition index)')
parser.add_argument("--pretrained", type=str, default="./moco_v2_800ep_pretrain.pth.tar")

args = parser.parse_args()


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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')


partitions_file = torch.load('partitions_hash_mean_MS-COCO_'+str(args.num_partitions)+'.pth')
partitions = partitions_file['idx']
#means = partitions_file['mean']
#stds = partitions_file['std']

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def eval(model, data_loader, device):

    model.eval()
    precision = 0
    recall = 0
    precision = 0
    count = 0
    for i, (input, target) in enumerate(data_loader):
        count+=1
        # target 1,-1
        if len(target.shape) == 3:
            targ = target.max(dim=1)[0].to(device)
        else:
            targ = F.relu(target).to(device)
        with torch.no_grad():
            # hard classifier: => >0 ,scorer sigmoid
            output = model(input.to(device))
            #print(torch.argmax(output,dim = 1))
            #print(targ)
            #print(torch.argmax(output, dim=1))
            topk_indices = torch.topk(output, k=10, dim=1)[1]
            #print(topk_indices)
            # create binary mask based on top-k indices
            mask = torch.zeros_like(output)
            mask.scatter_(1, topk_indices, 1)
            #print(mask)
            #print(targ)
            #rint(np.sum(output*targ))
            #print(targ,output)
            #print(np.sum(output*targ))
            precision +=(((targ* mask).sum().item())/10)
            recall+= ((targ* mask).sum().item()/targ.sum().item())
            #print(torch.eq(targ, mask).sum().item(), torch.eq(targ, mask).sum().item())
            #print(precision,recall)
    #print(precision,recall)
    print("=====DEVICE: ", device, "Precision:",precision/count,flush=True)
    print("=====DEVICE: ", device, "Recall:",recall/count,flush=True)
    return 1


def train(model, train_loader, optimizer, device):
    #print("model is on:",next(model.parameters()).get_device())
    start = timeit.default_timer()

    loss_func = AsymmetricLossOptimized()
    #loss_func = torch.nn.BCEWithLogitsLoss()
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
        #target[:,0]=0
        output = model(input.to(device))
        loss = loss_func(output, target.float())
        loss.backward()
        optimizer.step()
        tot += loss.item()

    #scheduler.step()

    #comm.Barrier()


    #print("DEVICE: ", device, " Train Loss tot: ", tot,flush=True)
    #print("DEVICE: ", device,"Train Loss tot: ", tot, file=fi,flush=True)
    
    #Your statements here

    stop = timeit.default_timer()

    #print("DEVICE: ", device,'Time: ', stop - start,flush=True)  


    return 1
if __name__ == "__main__":
    """
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    """
    print(rank)
    args.start_partition = (rank%2)*(int(args.num_partitions/2))
    args.num_partition_range = int(args.num_partitions/2)
    index = rank
    device = torch.device("cuda:"+str(index+5))
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
        transforms_train = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            CutoutPIL(cutout_factor=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transforms_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        instances_path_val = os.path.join(
            "./COCO", "annotations/instances_val2014.json"
        )
        val_data_path = os.path.join("./COCO", "val2014")
        val_dataset = CocoDetection(
            val_data_path,
            instances_path_val,
            transforms_test,
        )
        idxs=np.arange(len(val_dataset))
        np.random.shuffle(idxs)
        idxs=idxs[:int(1000)]
        val_dataset = torch.utils.data.Subset(val_dataset, idxs)
        instances_path_train = os.path.join(
            "./COCO", "annotations/instances_train2014.json"
        )
        train_data_path = os.path.join("./COCO", "train2014")
        trainset = CocoDetection(
            train_data_path,
            instances_path_train,
            transforms_train,
        )
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4)
        trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset,part_indices), batch_size=16, shuffle=True,num_workers=4)
        model = models.__dict__['resnet50']()
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 80)
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

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

        print("model created!",index)
        print(device)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=curr_lr, weight_decay=0.001)

    # Training
        for epoch in range(10):
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




