from PIL import Image
import os
from tqdm import tqdm

def resize_im(fp,im_size=(512,512)):
    for root,dirs, names in tqdm(os.walk(fp)):
        for name in tqdm(names):
            print(root.split('/')[-1])
            im_name = os.path.join(root,name)
            im = Image.open(im_name)
            im = im.resize(im_size)
            im.save(im_name)

base_train = os.path.join('data','train')
base_test = os.path.join('data','test')

resize_im(base_train,im_size=(512,512))
resize_im(base_test,im_size=(512,512))