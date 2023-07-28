#Install/import packages
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.metrics import categorical_accuracy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import *
from keras.layers import BatchNormalization
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from skimage import io
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sn
from sklearn.preprocessing import LabelBinarizer
import math
from sklearn.metrics import RocCurveDisplay
from keras.preprocessing import image
#from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def prepare_data():
    # images are 48x48
    # N = 35887
    Y = []
    X = []
    first = True
    # Modify for your working directory
    for line in open('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\icml_face_data.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[2].split()])

    #X, Y = np.array(X)/255, np.array(Y)
    X, Y = np.array(X), np.array(Y)
    X = X/255
    N, D = X.shape
    X = X.reshape(N, 48, 48, 1)
    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    num_class = len(set(Y))
    y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
    y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
    return X_train, X_test, y_train, y_test

def build_model():
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model():
    #Model
    path_model='model1_filter.h5'

    tf.keras.backend.clear_session()
    model=build_model()
    tf.keras.backend.set_value(model.optimizer.lr,1e-3)

    X_train, X_test, y_train, y_test = prepare_data()

    #fit the model
    h=model.fit(x=X_train,     
                y=y_train, 
                batch_size=64, 
                epochs=20, 
                verbose=1, 
                validation_data=(X_test,y_test),
                shuffle=True,
                callbacks=[
                    ModelCheckpoint(filepath=path_model),
                ]
                )
    return model

def load_model():
    model = build_model()
    path_checkpoint = "C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models\\CNN\\model1_filter.h5"
    model.load_weights(path_checkpoint)
    return model

def model_analysis():
    X_train, X_test, y_train, y_test = prepare_data()
    model = build_model()
    path_checkpoint = "C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models\\CNN\\model1_filter.h5"
    model.load_weights(path_checkpoint)
    y_pred=model.predict(X_test)
    y_pred_rounded = np.around(y_pred)
    #Accuracy score
    accuracy = accuracy_score(y_test, y_pred_rounded)
    #Validation Accuracy 
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("\n Best validation accuracy: %.2f%%" % (scores[1]*100))

    #Average Precision, Recall, F1-measure
    CNN_avg_precisionR = average_precision_score(y_test, y_pred_rounded)
    CNN_recallR = recall_score(y_test, y_pred_rounded, average='weighted')
    CNN_f1R = f1_score(y_test, y_pred_rounded, average='weighted')
    print(f'Average precision for CNN: {CNN_avg_precisionR}')
    print(f'Recall for CNN: {CNN_recallR}')
    print(f'F1-measure for CNN: {CNN_f1R}')

    cm = multilabel_confusion_matrix(y_test, y_pred_rounded)
    for i in range(len(cm)):
        print(f'Confusion matrix for label {i}:')
        print(cm[i])
    #ROC
    #Run Binarizer
    CNN_onehot, CNN_score = binarize(model, X_train, X_test, y_train, y_test, y_pred)
    #ROC Curves
    roc_curve([model], [CNN_onehot], [CNN_score])

# Binarize
#def binarize(model, tr_features, te_features, y_train, y_test):
def binarize(model, tr_features, te_features, y_train, y_test, y_pred):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    #y_score = model.fit(tr_features, y_train).predict_generator(te_features)
    y_score = y_pred
    return y_onehot_test, y_score

# Plot ROC curve
def roc_curve(models, onehot_tests, scores):
    # plotting parameters
    cols = 3
    linewidth = 1
    pos_label = 0  # mean 0 belongs to positive class
    rows = math.ceil(7 / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows*6))
    model_names = {0:'CNN', 1:'RF', 2:'Ada'}

    for i in range(7):
        for m in range(len(models)):
            model_name = model_names[m]
            display = RocCurveDisplay.from_predictions(onehot_tests[m][:, i],
                                                       scores[m][:, i],
                                                       name=f"{model_name}",
                                                       linewidth=linewidth,
                                                       ax=axs[i // cols, i % cols],)

        axs[i // cols, i % cols].plot([0, 1], [0, 1], linewidth=linewidth, linestyle=":")
        axs[i // cols, i % cols].set_title(i)
        axs[i // cols, i % cols].set_xlabel("False Positive Rate")
        axs[i // cols, i % cols].set_ylabel("True Positive Rate")
    plt.tight_layout(pad=2.0)  # spacing between subplots
    plt.show()

def emotion_prediction(model, img):
    objects = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    m=0.000000000000000000001
    custom = model.predict(img)
    a=custom[0]
    ind = 0
    for i in range(0,len(a)):
        if a[i]>m:
            m=a[i]
            ind=i
    return objects[ind]

# model_analysis()