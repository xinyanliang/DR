from keras.callbacks import LearningRateScheduler,TensorBoard,ModelCheckpoint

def scheduler128(epoch):
    lr = 0.1
    if epoch<150:
        lr = 0.003
    else:
        lr = 0.0003
    return lr

def scheduler256(epoch):
    lr = 0.1
    if epoch<150:
        lr = 0.003
    else:
        lr = 0.0003
    return lr
def scheduler512(epoch):
    lr = 0.1
    if epoch<150:
        lr = 0.3
    elif epoch<220:
        lr = 0.03
    else:
        lr = 0.0003
    return lr
