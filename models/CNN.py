import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd

class CNN:
    def __init__(self):
        np.random.seed(10)
        (self.x_Train, self.y_Train), (self.x_Test, self.y_Test) = mnist.load_data()
        #print('train data = ', len(x_train_image))
        #print('test data = ', len(x_test_image))
        self.model = Sequential()
        
    def run(self):
        # plot_images_labels_prediction(x_test_image, y_test_label, [], 0, 10)
        # print('x_train_image: ', x_train_image.shape)
        # print('y_train_label: ', y_train_label.shape)
        x_Train4D = self.x_Train.reshape(self.x_Train.shape[0], 28, 28, 1).astype('float32')
        x_Test4D = self.x_Test.reshape(self.x_Test.shape[0], 28, 28, 1).astype('float32')
        # print('x_train: ', x_Train.shape)
        # print('x_test: ', x_Test.shape)
        # print(x_train_image[0])
        x_Train4D_normalize = x_Train4D / 255
        x_Test4D_normalize = x_Test4D / 255
        # print( y_train_label[:5] )
        y_TrainOneHot = np_utils.to_categorical(self.y_Train)
        y_TestOneHot = np_utils.to_categorical(self.y_Test)
        # print(y_Train_OneHot[:5])
        # plot_image(x_train_image[0])
        # print(y_train_label[0])
        self.model.add(Conv2D(
            filters = 16,
            kernel_size = (5,5),
            padding = 'same',
            input_shape = (28,28,1),
            activation = 'relu'
            ))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Conv2D(
            filters = 36,
            kernel_size = (5,5),
            padding = 'same',
            activation = 'relu'
            ))
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))
        print(self.model.summary())
        self.model.compile(
            loss = 'categorical_crossentropy',
            optimizer = 'adam',
            metrics = ['accuracy'])
        train_history = self.model.fit(
            x = x_Train4D_normalize,
            y = y_TrainOneHot,
            validation_split = 0.2,
            epochs = 10,
            batch_size = 2300,
            verbose = 2)
        self.show_train_history(train_history, 'acc', 'val_acc')
        # show_train_history(train_history, 'loss', 'val_loss')
        scores = self.model.evaluate(x_Test4D_normalize, y_TestOneHot)
        print()
        print('accuracy = ', scores[1])
        prediction = self.model.predict_classes(x_Test4D_normalize)
        print()
        print()
        self.plot_images_labels_prediction(self.x_Test, self.y_Test, prediction, idx = 340)
        mtx = pd.crosstab(self.y_Test, 
            prediction,
            colnames = ['predict'],
            rownames = ['label'])
        print(mtx)
        print()
        df = pd.DataFrame({'label': self.y_Test, 'predict': prediction})
        print( df[:2] )

    def plot_image(self, image):
        fig = plt.gcf()
        fig.set_size_inches(2,2)
        plt.imshow(image, cmap='binary')
        plt.show()

    def plot_images_labels_prediction(self, images, labels, prediction, idx, num=10):
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

    def show_train_history(self, train_history, train, validation):
        plt.plot(train_history.history[train])
        plt.plot(train_history.history[validation])
        plt.title('Train History')
        plt.ylabel(train)
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc = 'upper left')
        plt.show()