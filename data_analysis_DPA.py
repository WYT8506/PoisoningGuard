import torch
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
from src.models import create_model
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
parser = argparse.ArgumentParser(description="ASL MS-COCO Inference on a single image")

parser.add_argument("--model_name", type=str, default="tresnet_xl")
parser.add_argument("--model_path", type=str, default="voc_asl.pth")
parser.add_argument("--input_size", type=int, default=448)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--dataset_type", type=str, default="NUS-WIDE")
parser.add_argument("--partitions", type=int, default=65)
parser.add_argument("--begin", type=float, default=0)
parser.add_argument("--end", type=float, default=2)
parser.add_argument("--T", type=int, default=100)

parser.add_argument("--N", type=int, default=100)
parser.add_argument("--M", type=int, default=500)
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--alpha", type=float, default=0.001)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--k_prime", type=int, default=1)
parser.add_argument("--record", type=str, default="record.txt")

args = parse_args(parser)




def eval_RS(model, data_loader, class_num,device):
    model.eval()

    # hyperameters
    n = args.N  # randomized smoothing iterations
    sigma = args.sigma  # co-variation of guassian
    alpha = args.alpha  # confidence
    k = args.k  # smoothing output num
    k_prime = args.k_prime  # hard output num

    precision, recall, c_precision, c_recall = 0, 0, 0, 0
    M = 0
    c_num = [0 for i in range(args.T + 10)]
    c_num2 = [0 for i in range(args.T + 10)]
    p_dom = 0
    r_dom = 0
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
            output = model(input.to(device))
            topk_indices = torch.topk(output, k=10, dim=1)[1]
            #print(topk_indices)
            # create binary mask based on top-k indices
            mask = torch.zeros_like(output)
            mask.scatter_(1, topk_indices, 1)
            precision +=(((targ* mask).sum().item())/10)
            recall+= ((targ* mask).sum().item()/targ.sum().item())
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
    print("=====DEVICE: ", device, "Precision:",precision/count,flush=True)
    print("=====DEVICE: ", device, "Recall:",recall/count,flush=True)

    return outputs,targets


def main():
    import os

    availble_gpus = list(range(torch.cuda.device_count()))

    print("Validation code for multi-label classification")

    torch.cuda.empty_cache()
    # setup model
    print("creating and loading the model...")
    
    if args.dataset_type == "PASCAL-VOC":  # ML-GCN
        args.num_classes = 20
        MEAN = [0, 0, 0]
        STD = [1, 1, 1]
        train_dataset = get_voc2007_train(args.input_size, MEAN, STD)
        val_dataset = get_voc2007_test(args.input_size, MEAN, STD)
    elif args.dataset_type == "MS-COCO":
        args.num_classes = 80
        args.do_bottleneck_head = False
        normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
        instances_path_val = os.path.join(
            "./COCO", "annotations/instances_val2014.json"
        )
        val_data_path = os.path.join("./COCO", "val2014")
        val_dataset = CocoDetection(
            val_data_path,
            instances_path_val,
            transforms.Compose(
                [
                    transforms.Resize((args.input_size, args.input_size)),
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
        args.num_classes = 81
        MEAN = [0, 0, 0]
        STD = [1, 1, 1]
        train_dataset = get_nuswide_train(args.input_size, MEAN, STD)
        val_dataset = get_nuswide_test(args.input_size, MEAN, STD)
    idxs=np.arange(len(val_dataset))
    np.random.shuffle(idxs)
    idxs=idxs[:int(500)]
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print("data loaded")
    
    all_outputs = []
    all_targets = []
    path = './checkpoints/NUS_nin_baseline_partitions_'+str(args.partitions)

    # get list of all filenames in directory
    filenames = os.listdir(path)
    filenames = filenames[0:50]

    # print all filenames
    for filename in filenames:
        print(filename)
    print(len(filenames))
    count = 0
    for filename in filenames:
        count+=1
        print(count)
        args.model_path = path+'/'+filename
        model = models.resnet18(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, args.num_classes)
        state = torch.load(args.model_path, map_location="cpu")
        model.load_state_dict(state['net'])

        device = torch.device("cuda:6" if len(availble_gpus) > 0 else "cpu")

        model.to(device)
        outputs, targets = eval_RS(model, val_loader, args.num_classes,device)
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
    np.save('NUS_results_partitions_'+str(args.partitions)+'/all_outputs.npy', all_outputs)
    np.save('NUS_results_partitions_'+str(args.partitions)+'/all_targets.npy', all_targets)
if __name__ == "__main__":
    main()
