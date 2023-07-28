import cv2
import numpy as np
import deepface 
from keras.preprocessing import image
import sys
sys.path.append('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models\\CNN')
import CNN as cnn
sys.path.append('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models')
import evaluation
sys.path.append('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Preprocessing')
import DataPreprocessingDemo
sys.path.append('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models\\Paper')
import KhanNeuralNetwork as knn
# import yolov5
# Source
# https://github.com/serengil/deepface/blob/master/deepface/DeepFace.py
'''
@inproceedings{serengil2021lightface,
  title        = {HyperExtended LightFace: A Facial Attribute Analysis Framework},
  author       = {Serengil, Sefik Ilkin and Ozpinar, Alper},
  booktitle    = {2021 International Conference on Engineering and Emerging Technologies (ICEET)},
  pages        = {1-4},
  year         = {2021},
  doi          = {10.1109/ICEET53442.2021.9659697},
  url          = {https://doi.org/10.1109/ICEET53442.2021.9659697},
  organization = {IEEE}
}
'''
from deepface import DeepFace
# Source for cv2 video capture
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81

# Load the cascade - has to be absolute path
# otherwise it will error saying it cant open in read mode 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

#model = cnn.load_model()
khan = knn.load_model()
#svm = evaluation.get_svm()
while True:
    # Read the frame
    ret, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # Emotion Prediction
        # emotion_prediction = emotion_model.predict(  cropped_img  )
        try: 
            # Frame passed into function has to be BGR
            # emotion_analysis = DeepFace.analyze(frame, actions=['emotion'])
            # emotion_results = emotion_analysis[0]['dominant_emotion']
            fv = DataPreprocessingDemo.feature_extraction(frame)
            #emotion_results = evaluation.svm_predict(svm, fv)
            emotion_results = 'NULL'
            if(len(fv) != 0):
                emotion_results, confidence = knn.get_prediction(khan, fv)
           
            # img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (48, 48)) 
            # img = np.array(img.tolist())
            # img = 1/255 * img
            # img = img.reshape((48,48,1))
            # img = np.expand_dims(img, axis = 0)
            # emotion_results = cnn.emotion_prediction(model, img)
        except:
            emotion_results = "Error Occured"
        cv2.putText(frame, f"Emotion: {emotion_results}\nConfidence: {confidence}", (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Window Title', cv2.resize(frame,(1200,860),interpolation = cv2.INTER_CUBIC))
    
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

# Release the VideoCapture object
cap.release()
