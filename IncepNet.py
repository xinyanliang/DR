from sklearn.utils import class_weight
import numpy as np
import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
from keras.utils import multi_gpu_model,to_categorical

from model import  IncepNet
from load_DR import DR
from train_callback import scheduler128



os.environ["CUDA_VISIBLE_DEVICES"] = "5"
n_gpu = 1
cpus = 10
base_path = os.path.join('data','img256')

loss = 'categorical_crossentropy'
# loss = 'mse'
epochs = 200
batch_size = 32
im_size = (256,256,3)
im_s = (256,256)
output_shape = 5

(train_x,train_y),(test_x,test_y) = DR(base_path)
cla_weight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)
class_idx = np.unique(train_y)
class_weight = {i: cla_weight[i] for i in class_idx}

if loss is not 'mse':
    train_y = to_categorical(train_y,5)
    test_y = to_categorical(test_y,5)
    output_shape = 5

data_gen_args = dict(rescale=1./255,
                     rotation_range=20.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     #shear_range=0.2,
                     horizontal_flip=True,
                     fill_mode='constant')

train_datagen  = ImageDataGenerator(**data_gen_args )
train_generator  = train_datagen.flow(x = train_x,y=train_y,batch_size=batch_size)
nb_train_samples = train_x.shape[0]

test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow(x = test_x,y=test_y,batch_size=batch_size)
nb_test_samples = test_x.shape[0]


if n_gpu > 1:
    with tf.device('/cpu:0'):
        model = IncepNet(im_size,classes=output_shape)
        parallel_model = multi_gpu_model(model, gpus=n_gpu)
else:
    model = IncepNet(im_size, classes=output_shape)

print(model.summary())
print(class_weight)

best_weight = 'inception{epoch:04d}.h5'
modelcheck = ModelCheckpoint(best_weight, monitor='val_acc',
                 save_best_only=True,period=0)
lr_scheduler = LearningRateScheduler(scheduler128)
csv = CSVLogger('incepnet5.csv')
parallel_model = model
parallel_model.compile(optimizer='sgd',loss=loss,
                       metrics=['acc'])
parallel_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size,
    class_weight= class_weight,
    callbacks=[modelcheck,lr_scheduler,csv],
    use_multiprocessing=True,
    workers=cpus)

model.save_weights('incepnet5.h5')