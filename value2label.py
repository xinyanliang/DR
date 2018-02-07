import numpy as np
from sklearn.metrics import cohen_kappa_score,confusion_matrix,accuracy_score
from keras.models import load_model
from load_DR import DR_test

split = [0.5,1.5,2.5,3.5]
label = [0,1,2,3,4]
(test_x,test_y) = DR_test()
model = load_model('model.h5')

pre_value = model.predict(test_x)

pre_label = []
for i in pre_value:
    if i <= split[0]:
        pre_label.append(label[0])
    elif i <=split[1]:
        pre_label.append(label[1])
    elif i <=split[2]:
        pre_label.append(label[2])
    elif i <=split[3]:
        pre_label.append(label[3])
    else:
        pre_label.append(label[4])
pre_label = np.array(pre_label)
test_y = np.array(test_y)

kappa = cohen_kappa_score(test_y,pre_label)
conf_matrix = confusion_matrix(test_y,pre_label)
accuracy = (test_y,pre_label)

print('kappa:',accuracy)
print('accuracy:',accuracy)
print('conf_matrix:',conf_matrix)
