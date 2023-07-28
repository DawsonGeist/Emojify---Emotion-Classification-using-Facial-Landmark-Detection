import numpy as np
from PIL import Image
import imutils
from imutils import face_utils
import dlib
import cv2
import io
from deepface import DeepFace
import sys
sys.path.append('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Preprocessing')
import DataPreprocessingDemo


def convert_emotion_string(emotion_string):
    encoded_emotion = -1
    if emotion_string == 'sad':
        encoded_emotion = 4
    elif emotion_string == 'angry':
        encoded_emotion = 0
    elif emotion_string == 'surprise':
        encoded_emotion = 5
    elif emotion_string == 'fear':
        encoded_emotion = 2
    elif emotion_string == 'happy':
        encoded_emotion = 3
    elif emotion_string == 'disgust':
        encoded_emotion = 1
    elif emotion_string == 'neutral':
        encoded_emotion = 6
    return encoded_emotion

def deepface_comparison():
    x,y = DataPreprocessingDemo.read_data('C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Data\\icml_face_data.csv')
    y_pred = []
    picture_removed = []
    for count, sample in enumerate(x):
        image = DataPreprocessingDemo.form_image(sample)
        print("image: ",count)
        try:
            emotion_analysis = DeepFace.analyze(image, actions=['emotion'])
            # Get Most-likely estimate
            # emotion_results = emotion_analysis[0]['dominant_emotion']
            # y_pred.append(convert_emotion_string(emotion_results))
            # Get Prediction Probability
            y_partial = []
            y_partial.append(emotion_analysis[0]['emotion'].get('angry'))
            y_partial.append(emotion_analysis[0]['emotion'].get('disgust'))
            y_partial.append(emotion_analysis[0]['emotion'].get('fear'))
            y_partial.append(emotion_analysis[0]['emotion'].get('happy'))
            y_partial.append(emotion_analysis[0]['emotion'].get('sad'))
            y_partial.append(emotion_analysis[0]['emotion'].get('surprise'))
            y_partial.append(emotion_analysis[0]['emotion'].get('neutral'))
            y_pred.append(y_partial)
        except:
            print("Failed")
            picture_removed.append(count)
        # if count == NUM_IMAGES_TO_INPUT:
        #     break
    picture_removed.sort(reverse=True)
    for err in picture_removed:
        y.pop(err)
    save = np.array(y_pred)
    tr = np.array(y)
    np.savetxt("deepface_pred_proba_full.csv",save,delimiter=",", fmt='%s')
    np.savetxt("y_true_full.csv",tr,delimiter=",", fmt='%s')

deepface_comparison()