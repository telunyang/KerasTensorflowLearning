import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
np.random.seed(10)

(x_train_image, y_train_label), (x_test_image, y_test_label) = mnist.load_data()

#print('train data = ', len(x_train_image))
#print('test data = ', len(x_test_image))

def plot_image(image):
    fig = plt.gcf()
    fig.set_size_inches(2,2)
    plt.imshow(image, cmap='binary')
    plt.show()
    
# plot_image(x_train_image[0])
# print(y_train_label[0])

def plot_images_labels_prediction(images, labels, prediction, idx, num=10):
    fig = plt.gcf()
    fig.set_size_inches(12,14)
    if num > 25: num=25
    for i in range(0,num):
        ax = plt.subplot(5, 5, 1+i)
        ax.imshow(images[idx], cmap='binary')
        
        title = 'lable=' + str(labels[idx])
        if len(prediction) > 0:
            title += ',predict=' + str(prediction[idx])
        
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        idx += 1
    plt.show()
    
# plot_images_labels_prediction(x_test_image, y_test_label, [], 0, 10)

# print('x_train_image: ', x_train_image.shape)
# print('y_train_label: ', y_train_label.shape)

x_Train = x_train_image.reshape(60000, 784).astype('float32')
x_Test = x_test_image.reshape(10000, 784).astype('float32')

# print('x_train: ', x_Train.shape)
# print('x_test: ', x_Test.shape)

# print(x_train_image[0])

x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

# print( y_train_label[:5] )

y_Train_OneHot = np_utils.to_categorical(y_train_label)
y_Test_OneHot = np_utils.to_categorical(y_test_label)

# print(y_Train_OneHot[:5])


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()

model.add(Dense(
    units = 1000,
    input_dim = 784,
    kernel_initializer = 'normal',
    activation = 'relu'
    ))

model.add(Dropout(0.5))

model.add(Dense(
    units = 1000,
    input_dim = 784,
    kernel_initializer = 'normal',
    activation = 'relu'
    ))

model.add(Dropout(0.5))

model.add(Dense(
    units = 10,
    kernel_initializer = 'normal',
    activation = 'softmax'
    ))

print(model.summary())

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])
 
train_history = model.fit(x = x_Train_normalize,
                          y = y_Train_OneHot,
                          validation_split = 0.2,
                          epochs = 10,
                          batch_size = 200,
                          verbose = 2)
 
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plt.show()
 
show_train_history(train_history, 'acc', 'val_acc')
# show_train_history(train_history, 'loss', 'val_loss')
 
scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
print()
print('accuracy = ', scores[1])
 
prediction = model.predict_classes(x_Test)
print()
print()
plot_images_labels_prediction(x_test_image, y_test_label, prediction, idx = 340)
 
import pandas as pd
mtx = pd.crosstab(y_test_label, 
            prediction,
            colnames = ['predict'],
            rownames = ['label'])
 
print(mtx)
print()

df = pd.DataFrame({'label': y_test_label, 'predict': prediction})
print( df[:2] )