import torch
import os
from src.helper_functions.helper_functions import parse_args
from src.loss_functions.losses import AsymmetricLoss, AsymmetricLossOptimized
import argparse
from PIL import Image
import numpy as np
from statsmodels.stats.proportion import (
    proportion_confint,
    multinomial_proportions_confint,
)
from scipy.stats import norm
import pickle
#from src.models import create_model
from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from voc import *
from nus import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from probability import Get_Overlap
import random
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn
import pickle
import torchvision.models as models
import nuswide
parser = argparse.ArgumentParser(description="ASL MS-COCO Inference on a single image")

parser.add_argument("--model_name", type=str, default="tresnet_xl")
parser.add_argument("--model_path", type=str, default="voc_asl.pth")
parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--dataset_type", type=str, default="PASCAL-VOC")
parser.add_argument("--certification", type=str, default="dpa")
parser.add_argument("--begin", type=float, default=0)
parser.add_argument("--end", type=float, default=2)
parser.add_argument("--T", type=int, default=100)
parser.add_argument("--num_partitions", type=int, default=300)
parser.add_argument("--m", type=int, default=1000)
parser.add_argument("--N", type=int, default=100)
parser.add_argument("--M", type=int, default=500)
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--k_prime", type=int, default=1)
parser.add_argument("--record", type=str, default="record.txt")
parser.add_argument("--pretrained", type=str, default="./moco_v2_800ep_pretrain.pth.tar")

args = parse_args(parser)

def extract_feature(model, train_loader, device):
    embeddings_list = []
    target_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(train_loader):
            print(i,flush = True)
            output = model(input.to(device))
            #print(output.shape,flush = True)
            embeddings_list.append(output.squeeze().to("cpu"))
            target_list.append(target.to("cpu"))
    embeddings_list = torch.cat(embeddings_list, dim=0).contiguous()
    target_list = torch.cat(target_list, dim=0).contiguous()
    print(embeddings_list.shape, target_list.shape)
    return torch.utils.data.TensorDataset(embeddings_list.detach(),target_list.detach())
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
def eval_output(predictions,targets):
    precision = 0
    recall = 0
    precision = 0
    count = 0
    for i in range(len(predictions)):
        targ = targets[i]
        top_k = np.argsort(predictions,axis = -1)[i][::-1][0:10]
        arr = np.zeros(targ.shape)
        arr[top_k] = 1
        if targ.sum()!=0:
            precision +=(((targ* arr).sum())/10)
            recall+= ((targ* arr).sum()/targ.sum())
            count+=1
        #print(torch.eq(targ, mask).sum().item(), torch.eq(targ, mask).sum().item())
        #print(precision,recall)
    #print(precision,recall)
    print("Precision:",precision/count,flush=True)
    print("Recall:",recall/count,flush=True)
    return 1
def eval_RS(model, data_loader, class_num,device):
    model.eval()

  
    fi = open(args.record, "a")
    print(
        "N:%d L:%f R:%f k:%d k_prime:%d sigma:%f alpha:%f"
        % (args.N, args.begin, args.end, args.k, args.k_prime, args.sigma, args.alpha),
        file=fi,
    )
    fi.close()  # delete record and rewrite
    outputs = np.empty((0,args.num_classes))
    targets = np.empty((0,args.num_classes))
    recall = 0
    precision = 0
    count = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):
            count+=1
            # target 1,-1
            target =target.to(device)
            if len(target.shape) == 3:
                targ = target.max(dim=1)[0].to(torch.int)
            else:
                targ = F.relu(target).to(torch.int)
            #targ[:,0] =0
            output = model(input.to(device))
            #topk_indices = torch.topk(output, k=10, dim=1)[1]
            #print(topk_indices)
            # create binary mask based on top-k indices
            #mask = torch.zeros_like(output)
            #mask.scatter_(1, topk_indices, 1)
            #precision +=(((targ* mask).sum().item())/10)
            #recall+= ((targ* mask).sum().item()/targ.sum().item())
            output = output.to('cpu').numpy()
            targ = targ.cpu()
            outputs = np.concatenate((outputs,output), axis=0)
            targets = np.concatenate((targets,targ), axis=0)
            
            #print(mask)
            #print(targ)
            #rint(np.sum(output*targ))
            #print(targ,output)
            #print(np.sum(output*targ))

            #print(torch.eq(targ, mask).sum().item(), torch.eq(targ, mask).sum().item())
            #print(precision,recall)
    #print(precision,recall)

    return outputs,targets


def main():
    
    import os

    availble_gpus = list(range(torch.cuda.device_count()))
    device = torch.device("cuda:6" if len(availble_gpus) > 0 else "cpu")

    print("Validation code for multi-label classification")

    torch.cuda.empty_cache()
    # setup model
    print("creating and loading the model...")
    
    if args.dataset_type == "PASCAL-VOC":  # ML-GCN
        args.num_classes = 20
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
        train_dataset = get_voc2007_train(train_transform)
        val_dataset = get_voc2007_test(test_transform)
    elif args.dataset_type == "MS-COCO":
        args.num_classes = 80
        args.do_bottleneck_head = False
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        instances_path_val = os.path.join(
            "./COCO", "annotations/instances_val2014.json"
        )
        val_data_path = os.path.join("./COCO", "val2014")
        val_dataset = CocoDetection(
            val_data_path,
            instances_path_val,
            transforms.Compose(
                [
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        instances_path_train = os.path.join(
            "./COCO", "annotations/instances_train2014.json"
        )
        train_data_path = os.path.join("./COCO", "train2014")
        train_dataset = CocoDetection(
            train_data_path,
            instances_path_train,
            transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
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
        train_dataset = nuswide.NUSWIDEClassification('.', 'trainval',transform = train_transform)

        val_dataset = nuswide.NUSWIDEClassification('.', 'test',transform = test_transform)
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
    encoder.to(device)

    idxs=np.arange(len(val_dataset))
    np.random.shuffle(idxs)
    idxs=idxs[:int(1000)]
    val_dataset = torch.utils.data.Subset(val_dataset, idxs)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    val_feature_dataset = extract_feature(encoder, val_loader, device)
    val_feature_loader = torch.utils.data.DataLoader(
        val_feature_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    print("data loaded")
    
    all_outputs = []
    all_targets = []
    if args.dataset_type == "PASCAL-VOC":
        if args.certification == "bagging":
            path = './VOC_moco_models_m='+str(args.m)
        if args.certification == "dpa":
            path = './checkpoints/PASCAL-VOC_nin_baseline_PASCAL-VOC_moco_partitions_'+str(args.num_partitions)
    if args.dataset_type == "NUS-WIDE":
        if args.certification == "bagging":
            path = './NUS_moco_models_m='+str(args.m)
        if args.certification == "dpa":
            path = './checkpoints/NUS_nin_baseline_NUS_moco_partitions_'+str(args.num_partitions)
    if args.dataset_type == "MS-COCO":
        if args.certification == "bagging":
            path = './COCO_moco_models_m='+str(args.m)
        if args.certification == "dpa":
            path = './checkpoints/COCO_nin_baseline_COCO_moco_partitions_'+str(args.num_partitions)
    # get list of all filenames in directory
    filenames = os.listdir(path)
    filenames = filenames[0:1000]

    # print all filenames
    for filename in filenames:
        print(filename)
    print(len(filenames))
    count = 0
    for filename in filenames:
        count+=1
        print(count)
        args.model_path = path+'/'+filename
        fc = nn.Linear(num_ftrs, args.num_classes)
        # freeze all layers but the last fc
        #model = models.__dict__['resnet50']()
        #num_ftrs = model.fc.in_features
        #model.fc = nn.Linear(num_ftrs, args.num_classes)

        # load from pre-trained, before DistributedDataParallel constructor
        if os.path.isfile(args.model_path):
            print("=> loading checkpoint '{}'".format(args.model_path))
            if args.certification == "bagging":
                checkpoint = torch.load(args.model_path, map_location="cpu")
            if args.certification == "dpa":
                checkpoint = torch.load(args.model_path, map_location="cpu")['net']
            #print(checkpoint.keys())
            # rename moco pre-trained keys
            state_dict = checkpoint
            msg = fc.load_state_dict(state_dict, strict=False)
            #fc.weight.data.copy_(model.fc.weight.data)
            #fc.bias.data.copy_(model.fc.bias.data)
            print(msg)
            #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.model_path))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_path))

        fc.to(device)
        outputs, targets = eval_RS(fc, val_feature_loader, args.num_classes,device)
        eval_output(outputs,targets)
        #eval(model, val_loader, 0,device)
        print(outputs.shape)
        print(outputs)
        print(targets.shape)
        
        print(targets)
        all_outputs.append(outputs)
        all_targets.append(targets)
    all_outputs= np.array(all_outputs)
    all_targets= np.array(all_targets)
    print(all_outputs.shape)
    print(all_targets.shape)
    if args.dataset_type == "PASCAL-VOC":
        if args.certification == "bagging":
            if not os.path.exists('voc_moco_results_linear_m='+str(args.m)):
                os.makedirs('voc_moco_results_linear_m='+str(args.m))
            np.save('voc_moco_results_linear_m='+str(args.m)+'/all_outputs.npy', all_outputs)
            np.save('voc_moco_results_linear_m='+str(args.m)+'/all_targets.npy', all_targets)
        if args.certification =="dpa":
            if not os.path.exists('./voc_moco_results_linear_partitions='+str(args.num_partitions)):
                os.makedirs('./voc_moco_results_linear_partitions='+str(args.num_partitions))
            np.save('voc_moco_results_linear_partitions='+str(args.num_partitions)+'/all_outputs.npy', all_outputs)
            np.save('voc_moco_results_linear_partitions='+str(args.num_partitions)+'/all_targets.npy', all_targets)
    if args.dataset_type == "NUS-WIDE":
        if args.certification == "bagging":
            if not os.path.exists('nus_moco_results_m='+str(args.m)):
                os.makedirs('nus_moco_results_m='+str(args.m))
            np.save('nus_moco_results_m='+str(args.m)+'/all_outputs.npy', all_outputs)
            np.save('nus_moco_results_m='+str(args.m)+'/all_targets.npy', all_targets)
        if args.certification =="dpa":
            if not os.path.exists('./nus_moco_results_partitions='+str(args.num_partitions)):
                os.makedirs('./nus_moco_results_partitions='+str(args.num_partitions))
            np.save('nus_moco_results_partitions='+str(args.num_partitions)+'/all_outputs.npy', all_outputs)
            np.save('nus_moco_results_partitions='+str(args.num_partitions)+'/all_targets.npy', all_targets)
    if args.dataset_type == "MS-COCO":
        if args.certification == "bagging":
            if not os.path.exists('coco_moco_results_m='+str(args.m)):
                os.makedirs('coco_moco_results_m='+str(args.m))
            np.save('coco_moco_results_m='+str(args.m)+'/all_outputs.npy', all_outputs)
            np.save('coco_moco_results_m='+str(args.m)+'/all_targets.npy', all_targets)
        if args.certification == "dpa":
            if not os.path.exists('./coco_moco_results_partitions='+str(args.num_partitions)):
                os.makedirs('./coco_moco_results_partitions='+str(args.num_partitions))
            np.save('coco_moco_results_partitions='+str(args.num_partitions)+'/all_outputs.npy', all_outputs)
            np.save('coco_moco_results_partitions='+str(args.num_partitions)+'/all_targets.npy', all_targets)

if __name__ == "__main__":
    main()
