import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import random
def eval(predictions,targets):
    precision = 0
    recall = 0
    precision = 0
    count = 0
    for i in range(len(predictions)):
            targ = targets[i]
            top_k = np.argsort(predictions,axis = -1)[i][::-1][0:3]
            arr = np.zeros(targ.shape)
            arr[top_k] = 1
            if targ.sum()>0:
                precision +=(((targ* arr).sum())/3)
                recall+= ((targ* arr).sum()/targ.sum())
                count+=1
            #print(torch.eq(targ, mask).sum().item(), torch.eq(targ, mask).sum().item())
            #print(precision,recall)
    #print(precision,recall)
    print("Precision:",precision/count,flush=True)
    print("Recall:",recall/count,flush=True)
    return 1
def eval_ensemble(predictions,targets):
    precision = 0
    recall = 0
    precision = 0
    count = 0
    for i in range(len(predictions)):
        targ = targets[i]
        top_k = np.argsort(predictions,axis = -1)[i][::-1][0:3]
        #print(top_k)
        arr = np.zeros(targ.shape)
        arr[top_k] = 1
        if targ.sum()>0:
            precision +=(((targ* arr).sum())/3)
            recall+= ((targ* arr).sum()/targ.sum())
            count+=1
            #print(torch.eq(targ, mask).sum().item(), torch.eq(targ, mask).sum().item())
            #print(precision,recall)
    #print(precision,recall)
    precision =precision/count
    recall = recall/count
    f1 = 2*precision*recall/(precision+recall)
    print("Precision:",precision,flush=True)
    print("Recall:",recall,flush=True)
    print("f1:",f1,flush=True)
    return 1
dataset = "voc"
alpha = 0.001
c = 500
n_models =1000
k=3
if dataset == "coco":
    n = 82081
if dataset == "voc":
    n = 5011
if dataset == "nus":
    n = 134025
#n_p = 82081
k_b =1
T = 0
e = n - T
m =1000
if dataset == "coco":
    num_classes = 80
if dataset == "voc":
    num_classes =20
    m=100
if dataset == "nus":
    num_classes =81

fak = 1

def _lower_confidence_bound(NA: int, N: int, alpha: float) -> float:
    """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
    This function uses the Clopper-Pearson method.
    :param NA: the number of "successes"
    :param N: the number of total draws
    :param alpha: the confidence level
    :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
    """
    return proportion_confint(NA, N, alpha, method="beta")
def get_bounds(counts):
    lower_bounds = np.zeros((500,num_classes))
    upper_bounds = np.zeros((500,num_classes))
    for i in range(c):
        for j in range(num_classes):
            lower_bounds[i][j],upper_bounds[i][j] = _lower_confidence_bound(counts[i][j]*fak, n_models*fak, alpha/c)
    return lower_bounds, upper_bounds
"""
if dataset == "coco":
    all_targets = np.load('coco_moco_results_m=1000/all_targets.npy')
    all_outputs = np.load('coco_moco_results_m=1000/all_outputs.npy')
if dataset == "nus":
    all_targets = np.load('nus_moco_results_m=1000/all_targets.npy')
    all_outputs = np.load('nus_moco_results_m=1000/all_outputs.npy')
if dataset == "voc":
    all_targets = np.load('voc_moco_results_m=100/all_targets.npy')
    all_outputs = np.load('voc_moco_results_m=100/all_outputs.npy')
targets = all_targets[0]

print(targets.shape)
idxs = np.where(targets.sum(axis =-1)!=0)

# Print the loaded array
counts = np.zeros((targets.shape[0],num_classes))
print(all_outputs.shape)

for l in range(1,k_b+1):
    kb_largest = (np.argsort(all_outputs,axis = -1)[:,:,-l])
    for i in range(targets.shape[0]):
        for j in range(num_classes):
            counts[i][j] += np.count_nonzero(kb_largest[:,i] == j)
for i in range(10):
    visualize = i
    print("=======visualize the ",visualize,"th sample==========")
    print("top-10 of base classifiers:")
    print((np.argsort(all_outputs,axis = -1)[:,:,::-1])[0:50,visualize][:,0:10])
    print("top-10 of ensemble classifiers:")
    print(np.argsort(counts,axis = -1)[visualize][::-1][0:10])
    print("number of votes for each label:")
    print(counts[visualize])
    print("groud truth labels:")
    print(np.where(targets[visualize]>0))
    print(counts.shape)
print(counts.shape)
#eval_ensemble(counts,targets)
for id in range(10):
    print('base classifier ',id,'s performance:')
    eval(all_outputs[id],targets)
counts = counts[idxs][0:500]
targets = targets[idxs][0:500]
eval_ensemble(counts,targets)
"""

"DPA"
if dataset =="coco":
    all_targets = np.load('coco_moco_results_partitions=300/all_targets.npy')
    all_outputs = np.load('coco_moco_results_partitions=300/all_outputs.npy')
if dataset =="nus":
    all_targets = np.load('nus_moco_results_partitions=300/all_targets.npy')
    all_outputs = np.load('nus_moco_results_partitions=300/all_outputs.npy')
if dataset =="voc":
    all_targets = np.load('voc_moco_results_partitions=300/all_targets.npy')
    all_outputs = np.load('voc_moco_results_partitions=300/all_outputs.npy')
targets = all_targets[0]

print(all_outputs.shape)
idxs = np.where(targets.sum(axis =-1)!=0)
print(targets.shape)

# Print the loaded array
#print(np.argsort(all_outputs,axis = -1)[:,:,-2].shape)
counts = np.zeros((targets.shape[0],num_classes))
for l in range(1,k_b+1):
    kb_largest = (np.argsort(all_outputs,axis = -1)[:,:,-l])
    for i in range(targets.shape[0]):
        for j in range(num_classes):
            #print(kb_largest[:,i])
            counts[i][j] += np.count_nonzero(kb_largest[:,i] == j)
#counts = counts[idxs][0:500]
#targets = targets[idxs][0:500]
#all_outputs = all_outputs[:,idxs,:][0:500]
for i in range(10):
    visualize = i
    print("=======visualize the ",visualize,"th sample==========")
    print("top-10 of base classifiers:")
    print((np.argsort(all_outputs,axis = -1)[:,:,::-1])[0:50,visualize][:,0:10])
    print("top-10 of ensemble classifiers:")
    print(np.argsort(counts,axis = -1)[visualize][::-1][0:10])
    print("number of votes for each label:")
    print(counts[visualize])
    print("groud truth labels:")
    print(np.where(targets[visualize]>0))
    print(counts.shape)
print("ensemble classifier's performance:")
print(counts.shape)
print(all_outputs[id])
eval_ensemble(counts,targets)

for id in range(10):
    print('base classifier ',id,'s performance:')
    eval(all_outputs[id],targets)

counts = counts[idxs][0:500]
targets = targets[idxs][0:500]
