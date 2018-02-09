from model import  Net5
from train_callback import scheduler256
from load_DR import DR
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,CSVLogger
import tensorflow as tf
from keras.utils import multi_gpu_model,to_categorical
from keras.models import load_model



os.environ["CUDA_VISIBLE_DEVICES"] = "7"
n_gpu = 1
cpus = 50
base_path = os.path.join('data','img256')

# loss = 'categorical_crossentropy'
loss = 'mse'
epochs = 200
batch_size = 128
im_size = (256,256,3)
im_s = (256,256)
output_shape = 1
class_weight = {0:0.73,1:0.07,2:0.15,3:0.025,4:0.02}


(train_x,train_y),(test_x,test_y) = DR(base_path)
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
        model =  Net5(im_size,1)
        model.load_weights('model_128.h5')
        parallel_model = multi_gpu_model(model, gpus=n_gpu)
else:
    model = Net5(im_size, 1)
    model.load_weights('model_128.h5')

print(model.summary())

parallel_model = model
modelcheck = ModelCheckpoint('model_256_best.h5', monitor='val_loss',
                 save_best_only=True)
lr_scheduler = LearningRateScheduler(scheduler256)
csv = CSVLogger('mode256.csv')
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

model.save_weights('model_256.h5')