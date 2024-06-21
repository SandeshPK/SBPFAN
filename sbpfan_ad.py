import keras
import os, cv2
import numpy as np
import pandas as pd
from keras import Input
import tensorflow as tf
from keras.models import Model, load_model
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import keras.backend as K
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.layers import *
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

data_path='Generated Images - AZ_CN/' # Path for generated images
data_dir_list = ['0','1']
img_data_list=[]
labels = []
for dataset in data_dir_list:
    img_list=os.listdir(data_path+'/'+ dataset)
    print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    for img in img_list:
        img=cv2.imread(data_path + '/'+ dataset + '/'+ img )
        # img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(128,128))
        labels.append(dataset)
        img_data_list.append(img)
label=np.array(labels)
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data = img_data/255
print(img_data.shape)

plt.imshow(img_data[30], cmap = 'gray')
plt.show()


num_classes = 2 # Number of classes in your dataset
x,y = shuffle(img_data, label, random_state=42)


def SBPFAN(input, num_classes):
  sepconv11 = SeparableConv2D(32, kernel_size=(3, 3), padding= 'same',activation=LeakyReLU(alpha=0.02))(input)
  sepconv12 = SeparableConv2D(32, kernel_size=(3, 3), padding= 'same',activation=LeakyReLU(alpha=0.02))(sepconv11)

  sepconv21 = SeparableConv2D(32, kernel_size=(5, 5), padding= 'same',activation=LeakyReLU(alpha=0.02))(input)
  sepconv22 = SeparableConv2D(32, kernel_size=(5, 5), padding= 'same',activation=LeakyReLU(alpha=0.02))(sepconv21)

  # Pyramid-1
  conv1 = Conv2D(32,(3,3), dilation_rate = 1,activation = LeakyReLU(alpha=0.02))(sepconv12)
  x = Conv2D(32,(1,1), padding = 'same')(conv1)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a1 = Activation('sigmoid')(x)
  attention1 = Multiply()([a1, conv1])
  add1 = Add()([x, attention1])
  M1 = MaxPooling2D(pool_size = (3,3), padding = 'valid')(add1)
  F1 = Flatten()(M1)

  conv2 = Conv2D(32,(3,3), dilation_rate = 2,activation = LeakyReLU(alpha=0.02))(conv1)
  x = Conv2D(32,(1,1), padding = 'same')(conv2)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a2 = Activation('sigmoid')(x)
  attention2 = Multiply()([a2, conv2])
  add2 = Add()([x, attention2])
  M2 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add2)
  F2 = Flatten()(M2)

  conv3 = Conv2D(32,(3,3), dilation_rate = 3,activation = LeakyReLU(alpha=0.02))(conv2)
  x = Conv2D(32,(1,1), padding = 'same')(conv3)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a3 = Activation('sigmoid')(x)
  attention3 = Multiply()([a3, conv3])
  add3 = Add()([x, attention3])
  M3 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add3)
  F3 = Flatten()(M3)

  # Pyramid-2
  conv1 = Conv2D(32,(3,3), dilation_rate = 1,activation = LeakyReLU(alpha=0.02))(sepconv22)
  x = Conv2D(32,(1,1), padding = 'same')(conv1)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a1 = Activation('sigmoid')(x)
  attention1 = Multiply()([a1, conv1])
  add1 = Add()([x, attention1])
  M1 = MaxPooling2D(pool_size = (3,3), padding = 'valid')(add1)
  F4 = Flatten()(M1)

  conv2 = Conv2D(32,(3,3), dilation_rate = 2,activation = LeakyReLU(alpha=0.02))(conv1)
  x = Conv2D(32,(1,1), padding = 'same')(conv2)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a2 = Activation('sigmoid')(x)
  attention2 = Multiply()([a2, conv2])
  add2 = Add()([x, attention2])
  M2 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add2)
  F5 = Flatten()(M2)

  conv3 = Conv2D(32,(3,3), dilation_rate = 3,activation = LeakyReLU(alpha=0.02))(conv2)
  x = Conv2D(32,(1,1), padding = 'same')(conv3)
  x = Conv2D(32,(3,3), padding = 'same')(x)
  x = Conv2D(32,(1,1), padding = 'same')(x)
  a3 = Activation('sigmoid')(x)
  attention3 = Multiply()([a3, conv3])
  add3 = Add()([x, attention3])
  M3 = MaxPooling2D(pool_size = (3,3),padding = 'valid')(add3)
  F6 = Flatten()(M3)

  C = Concatenate()([F1,F2,F3,F4,F5,F6])
  D1  = Dense(128, activation = LeakyReLU(alpha=0.02))(C)
  D2  = Dense(64, activation = LeakyReLU(alpha=0.02))(D1)
  OUT  = Dense(num_classes, activation='softmax')(D2)
  model = Model(inputs= [input], outputs= OUT, name="BaseModel")
  return model

input = Input(shape=(128, 128, 3), name='input')
# Initialize your model
model = SBPFAN(input, num_classes)
model.summary()
# Initialize 10-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store the performance metrics from each fold
accuracy_scores = []
reports = []
hists = []

# Cross-validation loop
for train_index, test_index in kfold.split(x, y):
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    model.compile(optimizer= Adam(learning_rate = 0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size = 32, verbose = 1)
    hists.append(hist)

    pred = model.predict(X_test)
    y_pred = np.argmax(pred, axis = 1)
    y_true = np.argmax(y_test, axis = 1)

    report = classification_report(y_true, y_pred)
    print(report)
    if len(y_pred) != len(y_true):
        raise ValueError("Lengths of predictions and ground_truth lists must be the same.")

    correct_count = 0
    total_count = len(y_pred)

    for pred, truth in zip(y_pred, y_true):
        if pred == truth:
            correct_count += 1

    accuracy = correct_count / total_count * 100

    print(accuracy)
    accuracy_scores.append(accuracy)
    print(100*'*')

# Calculate and display the average performance across all folds
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
std_dev = np.std(accuracy_scores)

print(f"Mean Accuracy: {mean_accuracy:.2f} ± {std_dev:.2f}")