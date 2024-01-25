import numpy as np
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import random
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
    print("Precision:",precision/count,flush=True)
    print("Recall:",recall/count,flush=True)
    return 1

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
if dataset == "coco":
    all_targets = np.load('coco_moco_results_m=1000/all_targets.npy')
    all_outputs = np.load('coco_moco_results_m=1000/all_outputs.npy')
if dataset == "nus":
    all_targets = np.load('nus_moco_results_m=1000/all_targets.npy')
    all_outputs = np.load('nus_moco_results_m=1000/all_outputs.npy')
if dataset == "voc":
    all_targets = np.load('voc_moco_results_linear_m=100/all_targets.npy')
    all_outputs = np.load('voc_moco_results_linear_m=100/all_outputs.npy')
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
print(counts.shape)
#eval_ensemble(counts,targets)
for id in range(10):
    print('base classifier ',id,'s performance:')
    #eval(all_outputs[id],targets)
counts = counts[idxs][0:500]
targets = targets[idxs][0:500]
"""
for l in range(1,2+3):
    kb_largest = (np.argsort(all_outputs,axis = -1)[:,:,-l])
    for i in range(c):
        for j in range(num_classes):
            counts[i][j] += np.count_nonzero(kb_largest[:,i] == j)/3
print(counts)

random.seed(42)
how_many = 3
kb_largest = (np.argsort(all_outputs,axis = -1)[:,:,-how_many:-1])
print(kb_largest.shape)
print(kb_largest[random.randint(0,2),1])
for q in range(n_models):
    for i in range(c):
        for j in range(num_classes):
            counts[i][j] += kb_largest[q,i,random.randint(how_many-3,how_many-2)] == j
print(counts)
"""
"""
for i in range(n_models):
    a = all_outputs[i]
    b = np.zeros_like(all_outputs[i])
    b[np.arange(len(a)), a.argmax(axis = -1)] = 1
    print(np.argsort(a,axis = -1)[:,-2])
"""
    


lower,upper = get_bounds(counts)
#lower = lower-(1/(n**m))
#upper = upper+(1/(n**m))
for i in range(10):
    print(counts[i])
    print(lower[i])
    print(upper[i])
def get_V(target,upper):
    indexes = np.argsort(upper)[::-1]
    V =[]
    i = 0
    while len(V)<k-r+1:
        #print(np.sum(target))
        #print(upper)
        #print(target)
        if target[indexes[i]]==0: 
            V.append(upper[indexes[i]])
            i+=1
        else:
            i+=1
    return V
def get_U(target,lower):
    M = int(np.sum(target))
    indexes = np.argsort(lower)
    U =[]
    i = 0
    while len(U)<M-r+1:
        if target[indexes[i]]==1: 
            U.append(lower[indexes[i]])
            i+=1
        else:
            i+=1
    return U
def min_t(V):
    l = []
    for t in range(1,k-r+1+1):
        
        x = 0
        for i in range(t):
            x = x+V[k-r-i]
        #print(x)
        x =x*((n/n_p)**m)
        x = x+k_b*(1-(e/n_p)**m)
        x = x/t
        
        l.append(x)
    #print(l)
    return min(l)
def max_t(U):
    l = []
    for t in range(1,len(U)+1):
        x = 0
        for i in range(t):
            x = x+U[len(U)-1-i]
        #print(x)
        x = x-k_b*(1-(e/n)**m)
        x =x*((n/n_p)**m)
        
        x = x/t
        
        l.append(x)
    #print(l)
    return max(l)
def min_t_individual(V):
    l = []
    t = 1     
    x = 0
    for i in range(t):
        x = x+V[k-r-i]
    #print(x)
    x =x*((n/n_p)**m)
    x = x+1*(1-(e/n_p)**m)
    x = x/t
    
    l.append(x)
    #print("V:",min(l))
    #print(l)
    return min(l)
def max_t_individual(U):
    l = []
    t=1
    x = 0
    for i in range(t):
        x = x+U[len(U)-1-i]
    #print(x)
    x = x-1*(1-(e/n)**m)
    #print("Ux:",x)
    x =x*((n/n_p)**m)
    
    x = x/t
    
    l.append(x)
    #print("U:",max(l))
    #print(l)
    return max(l)
if dataset == "coco":
    RANGE = 121
if dataset =="nus":
    RANGE = 201
if dataset == "voc":
    RANGE = 121
"pg-bagging"
all_attack_types = [0]
pg_recalls =[]
pg_precisions = []
pg_f1=[]
Ts = []
for T in range(RANGE):
    T = T
    x = 0
    certified = 0
    all_max_r = np.zeros((c,len(all_attack_types)))
    for i in range(c):
        for type_ind in range(len(all_attack_types)):
            type_ = all_attack_types[type_ind]
            if type_ == 0:
                e = n-T
                n_p = n
            if type_ == 1:
                e = n
                n_p = n+T
            if type_ == 2:
                e = n-T
                n_p = n-T

            max_r = 1
            for r in range(1,int(min(k,np.sum(targets[i])))+1):
                r = r
                #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
                V =get_V(targets[i],upper[i])
                U = get_U(targets[i],lower[i])
                #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
                if (min(min_t(V),min_t_individual(V))<max(max_t_individual(U),max_t(U))):
                    max_r+=1
            all_max_r[i][type_ind] = max_r-1
    all_max_r =all_max_r.min(axis = -1)
    total_recall = 0
    total_precision = 0
    total_f1=0
    for i in range(c):
        total_recall = total_recall+all_max_r[i]/np.sum(targets[i])
        total_precision = total_precision + all_max_r[i]/k
        total_f1 = total_f1+2*all_max_r[i]/(np.sum(targets[i])+k)
    print(T,"recall:", total_recall/c, "precision",total_precision/c)
    pg_recalls.append(total_recall/c)
    pg_precisions.append(total_precision/c)
    pg_f1.append(total_f1/c)
    Ts.append(T)
"bagging"
all_attack_types = [0]
bagging_recalls =[]
bagging_precisions = []
bagging_f1 = []
for T in range(RANGE):
    T = T
    x = 0
    certified = 0
    all_max_r = np.zeros((c,len(all_attack_types)))
    for i in range(c):
        for type_ind in range(len(all_attack_types)):
            type_ = all_attack_types[type_ind]
            if type_ == 0:
                e = n-T
                n_p = n
            if type_ == 1:
                e = n
                n_p = n+T
            if type_ == 2:
                e = n-T
                n_p = n-T

            max_r = 1
            for r in range(1,int(min(k,np.sum(targets[i])))+1):
                r = r
                #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
                V =get_V(targets[i],upper[i])
                U = get_U(targets[i],lower[i])
                #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
                if min_t_individual(V)<max_t_individual(U):
                    max_r+=1
            all_max_r[i][type_ind] = max_r-1
    all_max_r =all_max_r.min(axis = -1)
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    for i in range(c):
        total_recall = total_recall+all_max_r[i]/np.sum(targets[i])
        total_precision = total_precision + all_max_r[i]/k
        total_f1 = total_f1+2*all_max_r[i]/(np.sum(targets[i])+k)
    print("recall:", total_recall/c, "precision",total_precision/c)
    bagging_recalls.append(total_recall/c)
    bagging_precisions.append(total_precision/c)
    bagging_f1.append(total_f1/c)
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
"""
for id in range(10):
    print('base classifier ',id,'s performance:')
    eval(all_outputs[id],targets)
"""
counts = counts[idxs][0:500]
targets = targets[idxs][0:500]
def DPA_get_V(target,count):
    indexes = np.argsort(count)[::-1]
    
    V =[]
    idx = []
    i = 0
    while len(V)<k-r+1:
        #print(np.sum(target))
        #print(upper)
        #print(target)
        if target[indexes[i]]==0: 
            V.append(count[indexes[i]])
            idx.append(indexes[i])
            i+=1
        else:
            i+=1
    return V,np.array(idx)
def DPA_get_U(target,count):
    M = int(np.sum(target))
    indexes = np.argsort(count)
    U =[]
    idx = []
    i = 0
    while len(U)<M-r+1:
        if target[indexes[i]]==1: 
            U.append(count[indexes[i]])
            idx.append(indexes[i])
            i+=1
        else:
            i+=1
    return U,np.array(idx)
def DPA_min_t(V):
    l = []
    for t in range(1,k-r+1+1):
        
        x = 0
        for i in range(t):
            x = x+V[k-r-i]
        #print(x)
        x = x+min(k_b*T,t*T)
        x = x/t

        l.append(x)
    #print(l)
    "also return the t value where the minimum is reached for tie breaking"
    return min(l),np.max(np.where(np.array(l)==np.min(np.array(l)))[0]+1)
def DPA_max_t(U):
    l = []
    for t in range(1,len(U)+1):
        x = 0
        for i in range(t):
            x = x+U[len(U)-1-i]
        #print(x)
        x = x-min(k_b*T,t*T)
        #print(k_b*T,t*T)
        x = x/t
        l.append(x)
    #print(l)
    return max(l),np.max(np.where(np.array(l)==np.max(np.array(l)))[0]+1)
def DPA_min_t_naive(V):
    l = []
    for t in range(1,k-r+1+1):
        
        x = 0
        for i in range(t):
            x = x+V[k-r-i]
        #print(x)
        x = x+k_b*T
        x = x/t

        l.append(x)
    #print(l)
    return min(l)
def DPA_max_t_naive(U):
    l = []
    for t in range(1,len(U)+1):
        x = 0
        for i in range(t):
            x = x+U[len(U)-1-i]
        #print(x)
        x = x-k_b*T
        print(k_b*T,t*T)
        x = x/t
        l.append(x)
    #print(l)
    return max(l)

def DPA_min_t_individual(V):
    l = []
    for t in range(1,2):
        
        x = 0
        for i in range(t):
            x = x+V[k-r-i]
        #print(x)
        x = x+min(k_b*T,t*T)
        x = x/t

        l.append(x)
    #print(l)
    return min(l)
def DPA_min_t_individual_insertion(V):
    l = []
    for t in range(1,2):
        
        x = 0
        for i in range(t):
            x = x+V[k-r-i]
        #print(x)
        x = x+min(k_b*0,t*0)
        x = x/t

        l.append(x)
    #print(l)
    return min(l)
def DPA_max_t_individual(U):
    l = []
    for t in range(1,2):
        x = 0
        for i in range(t):
            x = x+U[len(U)-1-i]
        #print(x)
        x = x-min(k_b*T,t*T)
        #print(k_b*T,t*T)
        x = x/t
        l.append(x)
    #print(l)
    return max(l)

"pg-dpa"
if dataset == "coco":
    RANGE = 121
if dataset == "nus":
    RANGE = 201
if dataset == "voc":
    RANGE = 121
k=3
k_b = 1

if dataset =="coco":
    all_targets = np.load('coco_moco_results_partitions=300/all_targets.npy')
    all_outputs = np.load('coco_moco_results_partitions=300/all_outputs.npy')
if dataset =="nus":
    all_targets = np.load('nus_moco_results_partitions=300/all_targets.npy')
    all_outputs = np.load('nus_moco_results_partitions=300/all_outputs.npy')
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


counts = counts[idxs][0:500]
targets = targets[idxs][0:500]

dpa_recalls =[]
dpa_precisions = []
dpa_f1 = []
for T in range(RANGE):
    T = 2*T
    x = 0
    certified = 0
    all_max_r = np.zeros(c)
    for i in range(c):
        max_r = max(1,k-(num_classes-np.sum(targets[i])))
        for r in range(max(1,k-(num_classes-np.sum(targets[i]))+1),int(min(k,np.sum(targets[i])))+1):
            r = r
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            V,v_idx =DPA_get_V(targets[i],counts[i])
            U,u_idx = DPA_get_U(targets[i],counts[i])
            #print(V,v_idx)
            #print(DPA_min_t(V)[1])
            #min_v_idx =v_idx[len(v_idx)-DPA_min_t(V)[1]:len(v_idx)]
            #max_u_idx = u_idx[len(u_idx)-DPA_max_t(U)[1]:len(u_idx)]
            #print(min_v_idx)
            #print(V,U)
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            if DPA_min_t(V)[0]<DPA_max_t(U)[0]: #or (DPA_min_t(V)[0]==DPA_max_t(U)[0] and np.max(max_u_idx) < np.min(min_v_idx)):
                max_r+=1
        all_max_r[i] = max_r-1

    total_recall = 0
    total_precision = 0
    total_f1 = 0
    for i in range(c):
        total_recall = total_recall+all_max_r[i]/np.sum(targets[i])
        total_precision = total_precision + all_max_r[i]/k
        total_f1 = total_f1+2*all_max_r[i]/(np.sum(targets[i])+k)
    print("recall:", total_recall/c, "precision",total_precision/c)
    dpa_recalls.append(total_recall/c)
    dpa_precisions.append(total_precision/c)
    dpa_f1.append(total_f1/c)
"dpa"
dpa_ind_recalls =[]
dpa_ind_precisions = []
dpa_ind_f1 = []
Ts =[]
for T in range(RANGE):
    T =2*T
    x = 0
    certified = 0
    all_max_r = np.zeros(c)
    for i in range(c):
        max_r = max(1,k-(num_classes-np.sum(targets[i])))
        for r in range(max(1,k-(num_classes-np.sum(targets[i]))+1),int(min(k,np.sum(targets[i])))+1):
            r = r
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            V,v_idx =DPA_get_V(targets[i],counts[i])
            #min_v_idx =v_idx[np.where(V == np.min(V))[0]]
            U,u_idx = DPA_get_U(targets[i],counts[i])
            #max_u_idx = u_idx[np.where(U == np.max(U))[0]]

            #print(V,U)
            #print(r, get_V(targets[i],upper[i]),get_U(targets[i],lower[i]))
            if DPA_min_t_individual(V)<DPA_max_t_individual(U):#or (DPA_min_t_individual(V)==DPA_max_t_individual(U) and np.max(max_u_idx) < np.min(min_v_idx)):
                max_r+=1
        all_max_r[i] = max_r-1
    Ts.append(T)
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    for i in range(c):
        total_recall = total_recall+all_max_r[i]/np.sum(targets[i])
        total_precision = total_precision + all_max_r[i]/k
        total_f1 = total_f1+2*all_max_r[i]/(np.sum(targets[i])+k)
    print("recall:", total_recall/c, "precision",total_precision/c)
    dpa_ind_recalls.append(total_recall/c)
    dpa_ind_precisions.append(total_precision/c)
    dpa_ind_f1.append(total_f1/c)
Ts = np.arange(RANGE)
plt.figure(figsize = (7,4))
plt.plot( Ts,pg_recalls, label = 'PG-Bagging',color ="red")

#plt.plot( Ts,bagging_recalls, label = 'Bagging Recall',linestyle = '--',color ="b")
plt.plot( Ts,bagging_recalls, label = 'Bagging',linestyle = '--',color ="orange")
#plt.plot( Ts,dpa_recalls, label = 'Ours(DPA) Recall',linestyle = '-.',color ="b")
plt.plot( Ts,dpa_recalls, label = 'PG-DPA',linestyle = '-.',color ="b")
#plt.plot( Ts,dpa_ind_recalls, label = 'DPA Recall',linestyle = 'dotted',color ="b")
plt.plot( Ts,dpa_ind_recalls, label = 'DPA',linestyle = 'dotted',color ="g")
#plt.plot( Ts,dpa_ind_precisions, label = 'DPA Precisions',linestyle = '-.',color ="r")
plt.xlabel('T', fontsize=20)
plt.ylabel('Certified top-k recall', fontsize=20)
plt.grid()
plt.legend(loc='upper right',fontsize=18)
plt.xlim(0)
plt.xlim(xmax =4)
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
if dataset == "nus":
    plt.xticks([0,40,80,120,160,200])
if dataset == "coco":
    plt.xticks([0,20,40,60,80,100,120])
if dataset == "voc":
    plt.xticks([0,20,40,60,80,100,120])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
#plt.tight_layout()
plt.rcParams['figure.dpi'] = 1000
plt.tight_layout()

plt.savefig(dataset+'_recall.pdf', dpi=1000,bbox_inches='tight')
plt.figure(figsize = (7,4))
#plt.plot(Ts,recalls, label = 'Ours(Bagging) Recall',color ="b")
plt.plot( Ts,pg_precisions, label = 'PG-Bagging',color ="red")

#plt.plot( Ts,bagging_recalls, label = 'Bagging Recall',linestyle = '--',color ="b")
plt.plot( Ts,bagging_precisions, label = 'Bagging',linestyle = '--',color ="orange")
#plt.plot( Ts,dpa_recalls, label = 'Ours(DPA) Recall',linestyle = '-.',color ="b")
plt.plot( Ts,dpa_precisions, label = 'PG-DPA',linestyle = '-.',color ="b")
#plt.plot( Ts,dpa_ind_recalls, label = 'DPA Recall',linestyle = 'dotted',color ="b")
plt.plot( Ts,dpa_ind_precisions, label = 'DPA',linestyle = 'dotted',color ="g")
plt.xlabel('T', fontsize=20)
plt.ylabel('Certified top-k precision', fontsize=20)
plt.grid()
plt.legend(loc='upper right',fontsize=18)
plt.xlim(0)
plt.xlim(xmax =4)
if dataset == "nus":
    plt.xticks([0,40,80,120,160,200])
if dataset == "coco":
    plt.xticks([0,20,40,60,80,100,120])
if dataset =="voc":
    plt.xticks([0,20,40,60,80,100,120])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()


plt.savefig(dataset+'_precision.pdf', dpi=1000,bbox_inches='tight')
plt.rcParams['figure.dpi'] = 1000
plt.show()
#plt.title('k=10')
#plt.title('k=10')
plt.figure(figsize = (7,4))
#plt.plot(Ts,recalls, label = 'Ours(Bagging) Recall',color ="b")
plt.plot( Ts,pg_f1, label = 'PG-Bagging',color ="red")

#plt.plot( Ts,bagging_recalls, label = 'Bagging Recall',linestyle = '--',color ="b")
plt.plot( Ts,bagging_f1, label = 'Bagging',linestyle = '--',color ="orange")
#plt.plot( Ts,dpa_recalls, label = 'Ours(DPA) Recall',linestyle = '-.',color ="b")
plt.plot( Ts,dpa_f1, label = 'PG-DPA',linestyle = '-.',color ="b")
#plt.plot( Ts,dpa_ind_recalls, label = 'DPA Recall',linestyle = 'dotted',color ="b")
plt.plot( Ts,dpa_ind_f1, label = 'DPA',linestyle = 'dotted',color ="g")
plt.xlabel('T', fontsize=20)
plt.ylabel('Certified top-k f1-score', fontsize=20)
plt.grid()
plt.legend(loc='upper right',fontsize=18)
plt.xlim(0)
plt.xlim(xmax =4)
if dataset == "nus":
    plt.xticks([0,40,80,120,160,200])
else:
    plt.xticks([0,20,40,60,80,100,120])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0])
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()

plt.savefig(dataset+'_f1.pdf', dpi=1000,bbox_inches='tight')

plt.rcParams['figure.dpi'] = 1000
plt.show()