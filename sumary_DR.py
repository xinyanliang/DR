import os
from tqdm import tqdm
import numpy as np

def load(fp):
    im_sumary = [0]*5
    for root,dirs, names in tqdm(os.walk(fp)):
        for name in tqdm(names):
            idx = int(root.split('/')[-1])
            im_sumary[idx] +=1
    return im_sumary

base_train = os.path.join('data','train')
base_test = os.path.join('data','test')

train = load(base_train)
train_sum = np.sum(np.array(train))
test = load(base_test)
test_sum = np.sum(np.array(test))
print(train,train_sum,train/train_sum)
print(test,test_sum,test/test_sum)