import torch
import torchvision
import argparse
import numpy as  np
import PIL
import torchvision.transforms as transforms
import nuswide
import os
from voc import *
from src.helper_functions.helper_functions import AverageMeter, CocoDetection
parser = argparse.ArgumentParser(description='Partition Data')
parser.add_argument('--dataset', default="NUS-WIDE", type=str, help='dataset to partition')
parser.add_argument('--partitions', default=500, type=int, help='number of partitions')
args = parser.parse_args()
channels =3
if (args.dataset == "MS-COCO"):
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    instances_path_train = os.path.join(
        "./COCO", "annotations/instances_train2014.json"
    )
    train_data_path = os.path.join("./COCO", "train2014")
    data = CocoDetection(
        train_data_path,
        instances_path_train,
        transforms.Compose(
            [
                transforms.Resize((32, 32))
            ]
        ),
    )
    """
    idxs=np.arange(len(data))
    np.random.shuffle(idxs)
    idxs=idxs[:int(1000)]
    data = torch.utils.data.Subset(data, idxs)
    """
if (args.dataset == "NUS-WIDE"):
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    data = nuswide.NUSWIDEClassification('.', 'trainval',transform = transforms.Compose(
            [
                transforms.Resize((32, 32))
            ]
        ))
if (args.dataset == "PASCAL-VOC"):
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    transform = transforms.Compose(
            [
                transforms.Resize((32, 32))
            ]
        )
    data = get_voc2007_train(transform)

if (args.dataset == "gtsrb"):
	data = GTSRB('./data', train=True)
if (args.dataset != "gtsrb"):
	imgs, labels = zip(*data)
	finalimgs = torch.stack(list(map((lambda x: torchvision.transforms.ToTensor()(x)), list(imgs))))
	for_sorting = (finalimgs*255).int()
	intmagessum = for_sorting.reshape(for_sorting.shape[0],-1).sum(dim=1) % args.partitions
	
else:
	labels = [label for x,label in data]
	imgs_scaled = [torchvision.transforms.ToTensor() ( torchvision.transforms.Resize((32,32),interpolation=PIL.Image.BILINEAR )(image)) for image, y in data]
	#imgs_scaled = [torchvision.transforms.ToTensor() ( torchvision.transforms.Resize((48,48),interpolation=PIL.Image.BILINEAR )(PIL.ImageOps.equalize(image))) for image, y in data] # To use histogram equalization
	finalimgs =  torch.stack(list(imgs_scaled))
	intmagessum = torch.stack([(torchvision.transforms.ToTensor()(image).reshape(-1)*255).int().sum()% args.partitions for image, y in data])
	for_sorting =finalimgs


idxgroup = list([(intmagessum  == i).nonzero() for i in range(args.partitions)])
#force index groups into an order that depends only on image content  (not indexes) so that (deterministic) training will not depend initial indices
print("order:")
idxgroup = list([idxgroup[i][np.lexsort(((for_sorting[idxgroup[i]].reshape(idxgroup[i].shape[0],-1))).numpy().transpose())] for i in range(args.partitions) ])

idxgroupout = list([x.squeeze().numpy() for x in idxgroup])
#means = torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).mean(dim=1) for i in range(args.partitions) ]))
#stds =  torch.stack(list([finalimgs[idxgroup[i]].permute(2,0,1,3,4).reshape(channels,-1).std(dim=1) for i in range(args.partitions) ]))
out = {'idx': idxgroupout }
torch.save(out, "partitions_hash_mean_" +args.dataset+'_'+str(args.partitions)+'.pth')