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
from sklearn.metrics import precision_score
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

def convert_y(y):
    y_expanded = []
    for label in y:
        if label == 0:
            y_expanded.append([1,0,0,0,0,0,0])
        elif label == 1:
            y_expanded.append([0,1,0,0,0,0,0])
        elif label == 2:
            y_expanded.append([0,0,1,0,0,0,0])
        elif label == 3:
            y_expanded.append([0,0,0,1,0,0,0])
        elif label == 4:
            y_expanded.append([0,0,0,0,1,0,0])
        elif label == 5:
            y_expanded.append([0,0,0,0,0,1,0])
        elif label == 6:
            y_expanded.append([0,0,0,0,0,0,1])
    return y_expanded


def load_data():
    X = np.loadtxt('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\feature_matrix.csv', delimiter=',')
    y = np.loadtxt('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\feature_matrix_classes.csv', delimiter=',') 
    y = np.array(convert_y(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


# Source: https://arxiv.org/pdf/1812.04510.pdf
def build_model():
    model = Sequential()
    input_shape = (18,)
    model.add(keras.Input(shape=input_shape))
    model.add(Dense(100,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7, activation="softmax"))
    opt = Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model):
    X_train, X_test, y_train, y_test = load_data()
    #Model
    path_model='Khan_Weights1.h5'
    tf.keras.backend.clear_session()
    #fit the model
    h=model.fit(x=X_train,     
                y=y_train, 
                batch_size=64, 
                epochs=200, 
                verbose=1, 
                validation_data=(X_test,y_test),
                shuffle=True,
                callbacks=[
                    ModelCheckpoint(filepath=path_model),
                ]
                )
    return model

def load_model():
    m = build_model()
    m.load_weights('Khan_Weights1.h5')
    return m

def get_prediction(model, fv):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    fv = fv.reshape((1,18))
    output = model.predict(fv)
    max = 0
    index = 0
    for count, val in enumerate(output[0]):
        if val > max:
            max = val
            index = count
    return emotions[index], output[0][index]


def convert_multivalue_to_single_value(y_pred, y_test):
    y_pred_short = []
    index = 0
    for pred in y_pred:
        max = 0
        for count, val in enumerate(pred):
            if val > max:
                max = val
                index = count
        y_pred_short.append(index)
    
    y_test_short = []
    index = 0
    for pred in y_test:
        max = 0
        for count, val in enumerate(pred):
            if val > max:
                max = val
                index = count
        y_test_short.append(index)

    return y_pred_short, y_test_short



def analysis(knn, X_train, y_train, X_test, y_test):
    y_pred = knn.predict(X_test)

    y_pred2, y_test = convert_multivalue_to_single_value(y_pred, y_test)
    
    print(accuracy_score(y_test, y_pred2))
    CNN_avg_precisionR = precision_score(y_test, y_pred2, average='weighted')
    CNN_recallR = recall_score(y_test, y_pred2, average='weighted')
    CNN_f1R = f1_score(y_test, y_pred2, average='weighted')
    print(f'Average precision for DF: {CNN_avg_precisionR}')
    print(f'Recall for DF: {CNN_recallR}')
    print(f'F1-measure for DF: {CNN_f1R}')

    #ROC
    #Run Binarizer
    CNN_onehot, CNN_score = binarize(knn, X_train, X_test, y_train, y_test, y_pred)
    #ROC Curves
    roc_curve([knn], [CNN_onehot], [CNN_score])


# Plot ROC curve
def roc_curve(models, onehot_tests, scores):
    # plotting parameters
    cols = 3
    linewidth = 1
    pos_label = 0  # mean 0 belongs to positive class
    rows = math.ceil(7 / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows*6))
    model_names = {0:'Khan-NN'}

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

# Binarize
#def binarize(model, tr_features, te_features, y_train, y_test):
def binarize(model, tr_features, te_features, y_train, y_test, y_pred):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    #y_score = model.fit(tr_features, y_train).predict_generator(te_features)
    y_score = y_pred
    return y_onehot_test, y_score

def test():    
    X_train, X_test, y_train, y_test = load_data() 
    m = build_model()
    print(m.summary())
    # m = train_model(m)
    m.load_weights('Khan_Weights1.h5')
    # m.save_weights('Khan_Weights1.h5')
    analysis(m,X_train, y_train, X_test, y_test)

# test()