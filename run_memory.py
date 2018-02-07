from model import  Net5
from load_DR import DR
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.utils import multi_gpu_model,to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
n_gpu = 2
base_path = os.path.join('data')

# loss = 'categorical_crossentropy'
loss = 'mse'
epochs = 250
batch_size = 32
im_size = (512,512,3)
im_s = (512,512)
output_shape = 1


(train_x,train_y),(test_x,test_y) = DR()
if loss is not 'mse':
    train_y = to_categorical(train_y,5)
    test_y = to_categorical(test_y,5)
    output_shape = 5


train_datagen  = ImageDataGenerator(rescale=1/255,horizontal_flip=True)
train_generator  = train_datagen.flow(x = train_x,y=train_y,batch_size=batch_size)
nb_train_samples = train_x.shape[0]

test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow(x = test_x,y=test_y,batch_size=batch_size)
nb_test_samples = test_x.shape[0]


with tf.device('/cpu:0'):
    model = Net5(im_size,input_shape=output_shape)
    print(model.summary())
parallel_model  = multi_gpu_model(model,gpus=n_gpu)
modelcheck = ModelCheckpoint('model_best.h5', monitor='val_loss',
                 save_best_only=True)
parallel_model.compile(optimizer='adam',loss=loss,
                       metrics=['acc'])
parallel_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=nb_test_samples // batch_size,
    class_weight= {0:0.73,1:0.07,2:0.15,3:0.025,4:0.02},
    callbacks=[modelcheck])

model.save_weights('model.h5')