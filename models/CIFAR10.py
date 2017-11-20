from keras.datasets import cifar10
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from jieba import pool

class CIFAR10:
    def __init__(self):
        np.random.seed(10)
        (self.x_img_train, self.y_label_train), (self.x_img_test, self.y_label_test) = cifar10.load_data()
        print()
        print('train: ', len(self.x_img_train))
        print('test: ', len(self.x_img_test))
        self.label_dict = {
            0:'airplane',
            1:'automobile',
            2:'bird',
            3:'cat',
            4:'deer',
            5:'dog',
            6:'frog',
            7:'horse',
            8:'ship',
            9:'truck'
            }
        
        #建立 Keras 的 Sequential 模型
        self.model = Sequential()
    
    def plot_image_labels_prediction(self, images, labels, prediction, idx, num=10):
        fig = plt.gcf()
        fig.set_size_inches(12,14)
        if num > 25: num = 25
        for i in range(0, num):
            ax = plt.subplot(5,5,1+i)
            ax.imshow(images[idx], cmap='binary')
            title = str(i) + ',' + self.label_dict[ labels[i][0] ]
            if len(prediction) > 0:
                title += '=>' + self.label_dict[ prediction[i] ]
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]);ax.set_yticks([])
            idx+=1
        plt.show()
    
    def run(self):
        #self.plot_image_labels_prediction(self.x_img_train, self.y_label_train, [], 0)
        
        #標準化(介於 0-1 之間)
        x_img_train_normalize = self.x_img_train.astype('float32') / 255.0
        x_img_test_normalize = self.x_img_test.astype('float32') / 255.0
        
        #One-Hot編碼
        y_label_train_OneHot = np_utils.to_categorical(self.y_label_train)
        y_label_test_OneHot = np_utils.to_categorical(self.y_label_test)
        
        #建立卷積層1
        self.model.add(
            Conv2D(
                filters=32,
                kernel_size=(3,3),
                input_shape=(32,32,3),
                activation='relu',
                padding='same'
                )
            )
        
        #增加 Dropout，避免過度訓練
        self.model.add(Dropout(rate=0.25))
        
        #建立池化層1
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        #建立卷積層2
        self.model.add(
            Conv2D(
                filters=64,
                kernel_size=(3,3),
                activation='relu',
                padding='same'
                )
            )
        
        #增加 Dropout，避免過度訓練
        self.model.add(Dropout(rate=0.25))
        
        #建立池化層2
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        
        #建立平坦層
        self.model.add(Flatten())
        
        #增加 Dropout，避免過度訓練
        self.model.add(Dropout(rate=0.25))
        
        #建立隱藏層
        self.model.add(Dense(1024, activation='relu'))
        
        #增加 Dropout，避免過度訓練
        self.model.add(Dropout(rate=0.25))
        
        #建立輸出層
        self.model.add(Dense(10, activation='softmax'))
        
        print(self.model.summary())
        
obj = CIFAR10()
obj.run()
        