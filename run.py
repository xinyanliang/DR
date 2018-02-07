from model import  Net5, Net4
import os
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.utils import multi_gpu_model

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
base_path = os.path.join('data')
train_X_path = os.path.join(base_path,'train')
validate_X_path = os.path.join(base_path,'test')
test_X_path = os.path.join(base_path,'test')

epochs = 250
batch_size = 32

im_size = (512,512,3)
im_s = (512,512)
p = [0.5,1.5,2.5,3.5]

with tf.device('/cpu:0'):
    model = Net5(im_size,input_shape=5)


train_datagen  = ImageDataGenerator(rescale=1/255,
                            horizontal_flip=True)
train_generator  = train_datagen.flow_from_directory(
    train_X_path,classes=['0','1','2','3','4'],
    target_size=im_s, color_mode='rgb',
    class_mode='categorical', #sparse
    batch_size=batch_size, shuffle=True)
nb_train_samples = train_generator.samples


test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
    test_X_path,classes=['0','1','2','3','4'],
    color_mode='rgb',
    class_mode='categorical', #sparse
    batch_size=batch_size)
nb_test_samples = test_generator.samples

parallel_model  = multi_gpu_model(model,gpus=2)
parallel_model.compile(optimizer='adam',loss='categorical_crossentropy',
                       metrics=['acc'])
parallel_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size,
    class_weight= {0:0.73,1:0.07,2:0.15,3:0.025,4:0.02})

model.save_weights('first_try.h5')