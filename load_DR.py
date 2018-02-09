import os
from tqdm import tqdm
import cv2
import numpy as np

def load_data(fp):
    ims = []
    labels = []
    for root,dirs, names in tqdm(os.walk(fp)):
        for name in tqdm(names):
            im_name = os.path.join(root,name)
            im = cv2.imread(im_name)
            ims.append(im)
            labels.append(int(root.split('/')[-1]))
    ims = np.array(ims)
    labels = np.array(labels)
    return ims,labels


def DR(base=os.path.join('data')):
    base_train = os.path.join(base, 'train')
    base_test = os.path.join(base, 'test')
    train_x,train_y = load_data(base_train)
    test_x, test_y = load_data(base_test)
    print('Loading data down!!!!')

    return (train_x,train_y),(test_x,test_y)

def DR_test(base_test = os.path.join('data')):
    base_test = os.path.join(base_test, 'test')
    test_x, test_y = load_data(base_test)
    return (test_x,test_y)
