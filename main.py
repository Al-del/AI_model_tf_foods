import gc
import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import pandas as pd
from sklearn.model_selection import train_test_split
def load_image(image_path):
    # Read the image file
    image = tf.io.read_file(image_path)

    # Decode the image file to a tensor
    image = tf.image.decode_jpeg(image, channels=1)
    return image
from sklearn.utils import shuffle

path='train'
#Load Elim.npy
data = np.load('Elim.npy')

class load():
    def __init__(self,path):
        with tf.device('/CPU:0'):

            self.df=pd.read_csv(os.path.join(path,'_annotations.csv'))
            self.df = self.df.drop_duplicates(subset='filename').reset_index(drop=True)
            #Make X and Y arrays
            self.X=[]
            self.Y=[]
            #Make a list with all the unique classes
            self.classes=self.df['class'].unique()
            #Eliminate data from classes that are not in the test set
            for i in data:
                self.classes = np.delete(self.classes, np.where(self.classes == i))
            print(f"Classes: {len(self.classes)}")
            for i in range(len(self.df['filename'])):
                        #print(f"{self.df['filename'][i]} {self.df['class'][i]} i: {i}")
                        #load in X the image from the path
                        if self.df['class'][i] not in data:
                            self.X.append(load_image(os.path.join(path,self.df['filename'][i])))
                            #load in Y the class

                            self.Y.append(self.df['class'][i])
                            #print(f"i {i} {self.df['class'][i]}")
            #Make them np arrays
            self.X=np.array(self.X)
            print(f"Y: {len(self.Y[0])}")
            self.Y_df = pd.get_dummies(self.Y)
            self.Y_df = self.Y_df.astype(int)
            # Convert the DataFrame to a numpy array
            self.Y_ = self.Y_df.values
            # Print the class name for the 2911-th element
            #print(self.predict_class_name(self.Y_[7],self.X[7]))
            self.X, self.Y = shuffle(self.X, self.Y_)
    def predict_class_name(self,Y_,X):
                self.class_name = self.Y_df.columns[np.argmax(Y_)]
                plt.imshow(X)
                plt.title(self.class_name)
                plt.show()
                return self.class_name
    def shuffle(self):
        self.Y = np.array(self.Y_, dtype=tf.int32)  # Convert to tf.int32 tensor
        self.X, self.Y = shuffle(self.X, self.Y)
        return self.X, self.Y
obj_train=load('train')
obj_val=load('test')
print(f"Train: {len(obj_train.X)} {len(obj_train.classes.shape)}")
print(f"Val: {len(obj_val.X)} {len(obj_val.classes.shape)}")
for i in obj_train.Y:
    k=0
    for j in obj_val.Y:
        if obj_train.Y_df.columns[np.argmax(i)]==obj_val.Y_df.columns[np.argmax(j)]:
            k+=1
    print(f"{obj_train.Y_df.columns[np.argmax(i)]} {k}")
    if k==0:
        #Delete all the X and Y that are not in the test set
        print(f"Eliminating {obj_train.Y_df.columns[np.argmax(i)]}")
#Save Elim


def train():
    with tf.device('/CPU:0'):

            model_1=tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(640, 640, 1)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2,2),
                tf.keras.layers.Conv2D(2, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dense(107, activation='softmax')
            ])
            model_1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
            model_1.summary()
            history = model_1.fit(obj_train.X, obj_train.Y, epochs=10, batch_size=32, validation_data=(obj_val.X, obj_val.Y), verbose=1)
            model_1.save('model_1.h5')