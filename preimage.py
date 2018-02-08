from PIL import Image
import os
from tqdm import tqdm
import subprocess

def resize_im(src_fp,des_fp,im_size=(512,512)):
    for root,dirs, names in os.walk(src_fp):
        for name in names:
            im = Image.open(os.path.join(root,name))
            im = im.resize(im_size)
            im.save(os.path.join(des_fp,root.split('/')[-1],name))

img_size = [128,256,512]
for img_s in tqdm(img_size):
    src_train = os.path.join('data','train')
    src_test = os.path.join('data','test')
    des_train = os.path.join('data','img'+str(img_s),'train')
    des_test = os.path.join('data','img'+str(img_s),'test')
    if not os.path.exists(des_train):
        for i in range(5):
            subprocess.call(['mkdir','-p',os.path.join(des_train,str(i))])
    if not os.path.exists(des_test):
        for i in range(5):
            subprocess.call(['mkdir','-p',os.path.join(des_test,str(i))])
    resize_im(src_train,des_train,im_size=(i,i))
    resize_im(src_test,des_test,im_size=(i,i))