from keras.layers import Input,Dense,LeakyReLU,BatchNormalization,Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D,GlobalAvgPool2D
from keras.models import Model
from keras.regularizers import l2



base_filters = 32
decay_L2 = l2(5e-4)

def block(model,filters = [32,32],kernel_sizes = [5,3],
          strides = [2,2], max=True):
    for i,k in enumerate(kernel_sizes):
        model = Conv2D(filters=filters[i], kernel_size=(k, k),
                       strides=(strides[i],strides[i]),
                       kernel_regularizer=decay_L2,
                       padding='same',
                       kernel_initializer='he_normal'
                       # activation='relu'
                       )(model)
        model =  BatchNormalization()(model)
        model = Activation('relu')(model)
        # model = LeakyReLU(alpha=0.01)(model)
    if max:
        model = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(model)
    return model


def Net5(im_size,input_shape=1):
    input = Input(shape=(im_size))
    block1 = block(model=input,filters = [32,32],
                  kernel_sizes = [5,3],
                  strides = [2,1], max=True)
    block2 = block(model=block1, filters=[64, 64, 64],
                   kernel_sizes=[5,3,3],
                   strides=[2,1,1], max=True)
    block3 = block(model=block2, filters=[128, 128, 128],
                   kernel_sizes=[3, 3, 3],
                   strides=[1, 1, 1], max=True)
    block4 = block(model=block3, filters=[256, 256, 256],
                   kernel_sizes=[3, 3, 3],
                   strides=[1, 1, 1], max=True)
    block5 = block(model=block4, filters=[512, 512],
                   kernel_sizes=[3, 3],
                   strides=[1, 1], max=False)

    gavgpool = GlobalAvgPool2D()(block5)

    if input_shape==1:
        output = Dense(units=input_shape)(gavgpool)
    else:
        output = Dense(units=input_shape,activation='softmax')(gavgpool)

    model = Model(inputs=[input],outputs=[output])
    return model


def Net4(im_size,input_shape=1):
    input = Input(shape=(im_size))
    block1 = block(model=input,filters = [32,32],
                  kernel_sizes = [4,4],
                  strides = [2,1], max=True)
    block2 = block(model=block1, filters=[64, 64, 64],
                   kernel_sizes=[4,4,4],
                   strides=[2,1,1], max=True)
    block3 = block(model=block2, filters=[128, 128, 128],
                   kernel_sizes=[4, 4, 4],
                   strides=[1, 1, 1], max=True)
    block4 = block(model=block3, filters=[256, 256, 256],
                   kernel_sizes=[4, 4, 4],
                   strides=[1, 1, 1], max=True)
    block5 = block(model=block4, filters=[512, 512],
                   kernel_sizes=[4],
                   strides=[1], max=False)

    gavgpool = GlobalAvgPool2D()(block5)

    if input_shape==1:
        output = Dense(units=input_shape)(gavgpool)
    else:
        output = Dense(units=input_shape,activation='softmax')(gavgpool)

    model = Model(inputs=[input],outputs=[output])

    return model
