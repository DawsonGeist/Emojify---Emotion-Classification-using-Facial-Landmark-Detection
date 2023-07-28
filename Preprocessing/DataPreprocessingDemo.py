import numpy as np
from PIL import Image
import imutils
from imutils import face_utils
import dlib
import cv2
import io
from deepface import DeepFace

import FeatureFunctions

NUM_IMAGES_TO_INPUT = 100

def read_data(filepath):
    f = open(filepath, "r")
    f_lines = f.readlines()
    # one row is column headers
    size_rows = len(f_lines) - 1
    size_columns = len(process_line(f_lines[1])) - 2
    x = np.zeros((size_rows, size_columns), dtype=np.uint8)
    # y = np.zeros((size_rows, 1), dtype=np.uint8)
    y = []
    for count, line in enumerate(f_lines[1:]):
        print("Processing row ",count,"/",size_rows-1)
        raw_data = process_line(line)
        for count_2, pixel in enumerate(raw_data[2:]):
            x[count][count_2] = pixel
        #y[count] = raw_data[0]
        y.append(raw_data[0])
        # # FOR TESTING PURPOSES
        # if(count == NUM_IMAGES_TO_INPUT):
        #     break
    return x, y

def process_line(line):
    line = line.replace("\n","").replace("Training", "0").replace("PrivateTest", "1").replace("PublicTest", "1")
    line_split = line.split(",")
    extended_line_split = line_split[:2]
    extended_line_split.extend(line_split[2].split(" "))
    return extended_line_split

def form_image(pixel_array):
    # im = Image.new("L", (48,48), "white")
    # im.putdata(pixel_array)
    # data = np.asarray(bytearray(im.tobytes()))
    # im.show()
    cv2.imwrite('temp.jpg', np.reshape(pixel_array, (48,48)))
    image = cv2.imread('temp.jpg')
    #image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    image = imutils.resize(image, width=500)
    return image

def opencv_face_detection(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    cv2.imshow('Window Title', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def feature_extraction_test(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    win = dlib.image_window()
    dets = detector(image, 1)
    points = []
    for d in dets:
        shape = predictor(image, d)
        # Draw the face landmarks on the screen.
        # win.add_overlay(shape)
        # for i in range(0, 68):
        #     points.append(shape.part(i))
        feature_vector = FeatureFunctions.generate_feature_vector(shape)
        print(feature_vector)
        win.add_overlay(shape)
        # lipDetections = dlib.full_object_detection(d,points)
        #win.add_overlay(lipDetections)

    win.add_overlay(dets)
    win.set_image(image)
    win.wait_until_closed()

def feature_extraction(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("C:\\Users\\Dawso\\Desktop\\Graduate Study\\Spring 2023\\ICSI 536 Machine Learning\\Project\\Emojify\\Models\\shape_predictor_68_face_landmarks.dat")
    dets = detector(image, 1)
    feature_vector = []
    for d in dets:
        shape = predictor(image, d)
        feature_vector = FeatureFunctions.generate_feature_vector(shape)
    return feature_vector

'''
We have to remove the class labels carefully. The sample_index is going to always point to the sample in x. 
where as the feature_matrix_index is going to be offset because if a feature etraction fails. it has to not increment
its index because we did not add a feature vector for the image that failed.  
'''
def build_feature_matrix(x,y):
    feature_matrix = np.zeros((x.shape[0], 18))
    num_samples = x.shape[0]
    picture_removed = []
    feature_matrix_index = 0
    for sample_index, sample in enumerate(x):
        # REMOVE THIS
        # if sample_index > NUM_IMAGES_TO_INPUT:
        #     break
        print(f"Extracting Features for Sample {sample_index} / {num_samples}")
        temp_image = form_image(sample)
        sample_features = feature_extraction(temp_image)
        if len(sample_features) == 0:
            print(f"IMAGE {sample_index+1}: NO DETECTION - REMOVING IT")
            picture_removed.append(sample_index+1)
            # The feature extraction failed for sample. remove the sample and its class
            # y.pop(sample_index)
            # feature_matrix = np.delete(feature_matrix, sample_index, 0)
            feature_matrix_index-=1
        else:
            for feature_index, feature in enumerate(sample_features):
                feature_matrix[feature_matrix_index][feature_index] = feature[0]
        feature_matrix_index +=1
        
    # Now we are going to adjust y so that it holds the correct class labels for the remaining pictures
    picture_removed.sort(reverse=True)
    for err in picture_removed:
        y.pop(err-1)

    return feature_matrix, y, picture_removed

# x, y = read_data("icml_face_data.csv")
# print(x[0])
# print(type(x[0][0]))
# image = form_image(x[2])
#data, y, err = build_feature_matrix(x, y)
# y_numpy = np.array(y)
# err_numpy = np.array(err)
# print("Sample 0:")
# print(data[0])
# print("Sample 1:")
# print(data[1])
# print("Sample 2:")
# print(data[2])
# np.savetxt("feature_matrix.csv", data, delimiter=",", fmt='%s')
# np.savetxt("feature_matrix_classes.csv",y_numpy,delimiter=",", fmt='%s')
# np.savetxt("samples_removed.csv",err_numpy,delimiter=",", fmt='%s')
# image = form_image(x[0])
#feature_extraction(image)
#opencv_face_detection(image)


