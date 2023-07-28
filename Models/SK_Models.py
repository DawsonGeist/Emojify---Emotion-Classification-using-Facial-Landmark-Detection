import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dlib
import cv2
import os
import imutils
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
import math
from sklearn.ensemble import AdaBoostClassifier
from deepface import DeepFace

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 
predictor = dlib.shape_predictor("C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models\\shape_predictor_68_face_landmarks.dat") 

def load_data():
    # Load training set
    train = pd.read_csv('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\train.csv')
    train.head()
    # Load whole dataset
    whole = pd.read_csv('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\icml_face_data.csv')
    whole.head() 
    # Extract testing set
    test = pd.concat([whole[whole[' Usage'] == 'PublicTest'], whole[whole[' Usage'] == 'PrivateTest']])
    test = test.drop(columns=[' Usage'])
    test = test.rename(columns = {' pixels':'pixels'})
    test.head()   

    Xtr = str_mat(train)
    ytr = train.loc[:, "emotion"]
    Xte = str_mat(test)
    yte = test.loc[:, "emotion"]
    return Xtr, ytr, Xte, yte

# Convert pixels to a matrix
def str_mat(data):
    pixels = data.loc[:, "pixels"]
    matrix = np.empty([pixels.shape[0], 2304])
    for i in data.index:
        one_sample = np.array([int(p) for p in pixels[i].split()])
        matrix[i-data.index[0]] = one_sample
    return matrix

def save_temp(idx, matrix):
    # face = matrix[idx].reshape(48,48)
    # plt.imsave("temp"+str(idx)+".jpg", face)

    cv2.imwrite("temp"+str(idx)+".jpg", np.reshape(matrix[idx], (48,48)))
    image = cv2.imread("temp"+str(idx)+".jpg")
    image = imutils.resize(image, width=500)

def delete_temp(img_name):
    os.remove(img_name)

def feature_vec(my_image, my_faces):
    my_landmarks = []
    for (x, y, w, h) in my_faces:  
        cv2.rectangle(my_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        # Converting the OpenCV rectangle coordinates to Dlib rectangle  
        dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))  
        my_landmarks = np.matrix([[p.x, p.y] for p in predictor(my_image, dlib_rect).parts()])  
    return my_landmarks

def extract_landmarks():
    Xtr, ytr, Xte, yte = load_data()
    # Initialize
    Xtr_features = np.zeros((Xtr.shape[0], 68*2))
    face_detected_tr = []
    face_not_detected_tr = []

    for i in range(Xtr.shape[0]):
        # Save image
        save_temp(i, Xtr)

        # Read the image  
        temp_image = cv2.imread("temp"+str(i)+".jpg")
        # temp_gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image  
        temp_faces = faceCascade.detectMultiScale(  
        temp_image,  
        scaleFactor=1.05,  
        minNeighbors=4,  
        minSize=(10, 10),  
        flags=cv2.CASCADE_SCALE_IMAGE)
        # print("Found {0} faces!".format(len(temp_faces)))  

        # # Show detected face
        # for (x, y, w, h) in temp_faces:
        #     cv2.rectangle(temp_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # cv2.imshow('Window Title', temp_image)
        # cv2.waitKey(0); 
        # cv2.destroyAllWindows(); 
        # cv2.waitKey(1)

        # Get landmarks
        if len(temp_faces) > 0:
            # Get landmarks
            landmarks = np.array(feature_vec(temp_image, temp_faces))
            face_detected_tr.append(i)
        else:
            landmarks = np.zeros((1, 136))
            face_not_detected_tr.append(i)

        # Store landmarks
        Xtr_features[i] = landmarks.flatten()

        # Delete image
        delete_temp("temp"+str(i)+".jpg")
    print(len(face_detected_tr), "faces detected, ", len(face_not_detected_tr), "faces not detected.")

    Xtr_features = Xtr_features[face_detected_tr]
    ytr = np.array(ytr)
    ytr = ytr[face_detected_tr]

    # Initialize
    Xte_features = np.zeros((Xte.shape[0], 68*2))
    face_detected_te = []
    face_not_detected_te = []

    for i in range(Xte.shape[0]):
        # Save image
        save_temp(i, Xte)

        # Read the image  
        temp_image = cv2.imread("temp"+str(i)+".jpg")
        # temp_gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image  
        temp_faces = faceCascade.detectMultiScale(  
        temp_image,  
        scaleFactor=1.05,  
        minNeighbors=4,  
        minSize=(10, 10),  
        flags=cv2.CASCADE_SCALE_IMAGE)

        # Get landmarks
        if len(temp_faces) > 0:
            # Get landmarks
            landmarks = np.array(feature_vec(temp_image, temp_faces))
            face_detected_te.append(i)
        else:
            landmarks = np.zeros((1, 136))
            face_not_detected_te.append(i)

        # Store landmarks
        Xte_features[i] = landmarks.flatten()

        # Delete image
        delete_temp("temp"+str(i)+".jpg")

    print(len(face_detected_te), "faces detected, ", len(face_not_detected_te), "faces not detected.")
    Xte_features = Xte_features[face_detected_te]
    yte = np.array(yte)
    yte = yte[face_detected_te]

    return Xtr_features, Xte_features, Xtr, Xte, ytr, yte

# Dawson's landmarks[a][b] = Here feature_matrix[2*a+b]
def feature_distance(feature_matrix):
    distance_matrix = np.zeros((feature_matrix.shape[0], 18))
    for i in range(feature_matrix.shape[0]):
        distances = []
        # right_eye_height
        distances.append(feature_matrix[i][83] - feature_matrix[i][75] + feature_matrix[i][81] - feature_matrix[i][77])
        # left_eye_height
        distances.append(feature_matrix[i][95] - feature_matrix[i][87] + feature_matrix[i][93] - feature_matrix[i][89])
        # right_eye_width
        distances.append(feature_matrix[i][78] - feature_matrix[i][72])
        # left_eye_width
        distances.append(feature_matrix[i][90] - feature_matrix[i][84])
        # left_eyebrow_width
        distances.append(feature_matrix[i][42] - feature_matrix[i][34])
        # right_eyebrow_width
        distances.append(feature_matrix[i][52] - feature_matrix[i][44])
        # right_eyebrow_eye_distance
        distances.append((feature_matrix[i][75] - feature_matrix[i][39] + feature_matrix[i][77] - feature_matrix[i][41]) / 2)
        # left_eyebrow_eye_distance
        distances.append((feature_matrix[i][87] - feature_matrix[i][47] + feature_matrix[i][89] - feature_matrix[i][49]) / 2)
        # mouth_gap
        distances.append((feature_matrix[i][135] + feature_matrix[i][133] + feature_matrix[i][131] - feature_matrix[i][125] - feature_matrix[i][123] - feature_matrix[i][127])/3)
        # mouth_height_center_avg
        distances.append((feature_matrix[i][117] + feature_matrix[i][115] + feature_matrix[i][113] - feature_matrix[i][101] - feature_matrix[i][103] - feature_matrix[i][105])/3)
        # mouth_height_left_avg
        distances.append((feature_matrix[i][119] + feature_matrix[i][117] + feature_matrix[i][115] - feature_matrix[i][99] - feature_matrix[i][101] - feature_matrix[i][103])/3)
        # mouth_height_right_avg
        distances.append((feature_matrix[i][115] + feature_matrix[i][113] + feature_matrix[i][111] - feature_matrix[i][103] - feature_matrix[i][105] - feature_matrix[i][107])/3)
        # mouth_height_centerline_toplip
        distances.append((feature_matrix[i][97] + feature_matrix[i][109])/2 - feature_matrix[i][103])
        # mouth_height_centerline_bottomlip
        distances.append(feature_matrix[i][115] - (feature_matrix[i][97] + feature_matrix[i][109])/2)
        # lip_width
        distances.append(feature_matrix[i][108] - feature_matrix[i][96])
        # distance_between_nose_toplip
        distances.append(feature_matrix[i][103] - feature_matrix[i][67])
        # distance_between_left_eye_corner_left_lip_corner
        lell = np.array([feature_matrix[i][96] - feature_matrix[i][72], feature_matrix[i][97] - feature_matrix[i][73]])
        distances.append((lell[0]**2 + lell[1]**2)**(1/2))
        # distance_between_right_eye_corner_right_lip_corner
        rerl = np.array([feature_matrix[i][90] - feature_matrix[i][108], feature_matrix[i][109] - feature_matrix[i][91]])
        distances.append((rerl[0]**2 + rerl[1]**2)**(1/2))
        # Add to the matrix
        distance_matrix[i] = np.array(distances)
    return distance_matrix

def gbm(Xtr_features, Xte_features, Xtr_distances, Xte_distances, Xtr, ytr, Xte, yte):

    lightGBM = lgb.LGBMClassifier(random_state=0)
    lightGBM.fit(Xtr_features, ytr)
    ypr_lgbm = lightGBM.predict(Xte_features)
    # Check accuracy
    print('LightGBM using raw coordinates: {0:0.4f}'.format(accuracy_score(yte, ypr_lgbm)))

    lightGBM2 = lgb.LGBMClassifier(random_state=0)
    lightGBM2.fit(Xtr_distances, ytr)
    ypr_lgbm2 = lightGBM2.predict(Xte_distances)
    # Check accuracy
    print('LightGBM using distances: {0:0.4f}'.format(accuracy_score(yte, ypr_lgbm2)))
    CNN_avg_precisionR = precision_score(yte, ypr_lgbm2, average='weighted')
    CNN_recallR = recall_score(yte, ypr_lgbm2, average='weighted')
    CNN_f1R = f1_score(yte, ypr_lgbm2, average='weighted')
    print(f'Average precision for GBM: {CNN_avg_precisionR}')
    print(f'Recall for GBM: {CNN_recallR}')
    print(f'F1-measure for GBM: {CNN_f1R}')
    return lightGBM, lightGBM2


def configurations_RF(depths):
    accuracies = []
    for d in depths:
        randomForest = RandomForestClassifier(max_depth=d, random_state=0)
        randomForest.fit(Xtr_features, ytr)
        ypr_rf = randomForest.predict(Xte_features)
        accuracies.append(accuracy_score(yte, ypr_rf))
    return accuracies


def rf(Xtr_features, Xte_features, Xtr_distances, Xte_distances, Xtr, ytr, Xte, yte):
    randomForest = RandomForestClassifier(max_depth=19, random_state=0)
    randomForest.fit(Xtr_features, ytr)
    ypr_rf = randomForest.predict(Xte_features)
    # Check accuracy
    print('RF using raw coordinates: {0:0.4f}'.format(accuracy_score(yte, ypr_rf)))

    randomForest2 = RandomForestClassifier(max_depth=19, random_state=0)
    randomForest2.fit(Xtr_distances, ytr)
    ypr_rf2 = randomForest2.predict(Xte_distances)
    # Check accuracy
    print('RF using distances: {0:0.4f}'.format(accuracy_score(yte, ypr_rf2)))
    CNN_avg_precisionR = precision_score(yte, ypr_rf2, average='weighted')
    CNN_recallR = recall_score(yte, ypr_rf2, average='weighted')
    CNN_f1R = f1_score(yte, ypr_rf2, average='weighted')
    print(f'Average precision for RF: {CNN_avg_precisionR}')
    print(f'Recall for RF: {CNN_recallR}')
    print(f'F1-measure for RF: {CNN_f1R}')
    return randomForest, randomForest2


def configurations_ADA(estimators):
    accuracies = []
    for d in estimators:
        adaBoost = AdaBoostClassifier(n_estimators=d, random_state=0)
        adaBoost.fit(Xtr_features, ytr)
        ypr_ada = adaBoost.predict(Xte_features)    
        accuracies.append(accuracy_score(yte, ypr_ada))
    return accuracies

def ada(Xtr_features, Xte_features, Xtr_distances, Xte_distances, Xtr, ytr, Xte, yte):
    adaBoost = AdaBoostClassifier(n_estimators=125, random_state=0)
    adaBoost.fit(Xtr_features, ytr)
    ypr_ada = adaBoost.predict(Xte_features)
    # Check accuracy
    print('AdaBoost using raw coordinates: {0:0.4f}'.format(accuracy_score(yte, ypr_ada)))
    adaBoost2 = AdaBoostClassifier(n_estimators=125, random_state=0)
    adaBoost2.fit(Xtr_distances, ytr)
    ypr_ada2 = adaBoost2.predict(Xte_distances)
    # Check accuracy
    print('AdaBoost using distances: {0:0.4f}'.format(accuracy_score(yte, ypr_ada2)))
    #Average Precision, Recall, F1-measure
    CNN_avg_precisionR = precision_score(yte, ypr_ada2, average='weighted')
    CNN_recallR = recall_score(yte, ypr_ada2, average='weighted')
    CNN_f1R = f1_score(yte, ypr_ada2, average='weighted')
    print(f'Average precision for Ada: {CNN_avg_precisionR}')
    print(f'Recall for Ada: {CNN_recallR}')
    print(f'F1-measure for Ada: {CNN_f1R}')
    return adaBoost, adaBoost2


# Binarize
def binarize(model, tr_features, te_features, y_tr, y_te):
    label_binarizer = LabelBinarizer().fit(y_tr)
    y_onehot_test = label_binarizer.transform(y_te)
    y_score = model.fit(tr_features, ytr).predict_proba(te_features)
    return y_onehot_test, y_score

# Plot ROC curve
def roc_curve(models, onehot_tests, scores):
    # plotting parameters
    cols = 3
    linewidth = 1
    rows = math.ceil(7 / cols)
    emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows*6))
    model_names = {0:'LightGBM', 1:'RF', 2:'Ada'}

    for i in range(7):
        emotion_name = emotions[i]
        for m in range(len(models)):
            model_name = model_names[m]
            display = RocCurveDisplay.from_predictions(onehot_tests[m][:, i],
                                                       scores[m][:, i],
                                                       name=f"{model_name}",
                                                       linewidth=linewidth,
                                                       ax=axs[i // cols, i % cols],)

        axs[i // cols, i % cols].plot([0, 1], [0, 1], linewidth=linewidth, linestyle=":")
        axs[i // cols, i % cols].set_title(emotion_name)
        axs[i // cols, i % cols].set_xlabel("False Positive Rate")
        axs[i // cols, i % cols].set_ylabel("True Positive Rate")
    plt.tight_layout(pad=2.0)  # spacing between subplots
    plt.show()

def find_confidence_score(method,data):
    emotion_labels = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    confidence_scores = method.predict_proba(data)
    confidence_list = []
    for i in range(len(confidence_scores)):
        confidence_list.append(max(confidence_scores[i]))
        confidence_scores[i]
        max_index = np.argmax(confidence_scores[i])
        emotion_label = emotion_labels.get(max_index, "Unknown") 
        print(confidence_list[i],emotion_label,max_index)
    
    
    return confidence_list

Xtr_features, Xte_features, Xtr, Xte, ytr, yte = extract_landmarks()
Xtr_distances = feature_distance(Xtr_features)
Xte_distances = feature_distance(Xte_features)

lightGBM, lightGBM2 = gbm(Xtr_features, Xte_features, Xtr_distances, Xte_distances, Xtr, ytr, Xte, yte)
randomForest, randomForest2 = rf(Xtr_features, Xte_features, Xtr_distances, Xte_distances, Xtr, ytr, Xte, yte)
adaBoost, adaBoost2 = ada(Xtr_features, Xte_features, Xtr_distances, Xte_distances, Xtr, ytr, Xte, yte)

find_confidence_score(lightGBM, Xte_features)
find_confidence_score(lightGBM2, Xte_distances)
find_confidence_score(randomForest, Xte_features)
find_confidence_score(randomForest2, Xte_distances)
find_confidence_score(adaBoost, Xte_features)
find_confidence_score(adaBoost2, Xte_distances)

gbm_oh, gbm_sc = binarize(lightGBM, Xtr_features, Xte_features, ytr, yte)
rf_oh, rf_sc = binarize(randomForest, Xtr_features, Xte_features, ytr, yte)
ada_oh, ada_sc = binarize(adaBoost, Xtr_features, Xte_features, ytr, yte)

gbm_oh2,gbm_sc2 = binarize(lightGBM2, Xtr_distances, Xte_distances, ytr, yte)
rf_oh2, rf_sc2 = binarize(randomForest2, Xtr_distances, Xte_distances, ytr, yte)
ada_oh2, ada_sc2 = binarize(adaBoost2, Xtr_distances, Xte_distances, ytr, yte)

roc_curve([lightGBM, randomForest, adaBoost], [gbm_oh, rf_oh, ada_oh], [gbm_sc, rf_sc, ada_sc])
roc_curve([lightGBM2, randomForest2, adaBoost2], [gbm_oh2, rf_oh2, ada_oh2], [gbm_sc2, rf_sc2, ada_sc2])

