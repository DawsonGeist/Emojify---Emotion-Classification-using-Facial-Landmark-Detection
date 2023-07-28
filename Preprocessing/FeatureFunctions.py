import numpy as np

# Rectangle to Bounding Box
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
# Convert dlib shape object to numpy array
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68,2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

# get the total vertical distance between the top and
# bottom of the right eye 
def right_eye_height(landmarks):
    return landmarks[41][1] - landmarks[37][1] + landmarks[40][1] - landmarks[38][1]

# get the total vertical distance between the top and
# bottom of the left eye 
def left_eye_height(landmarks):
    return landmarks[47][1] - landmarks[43][1] + landmarks[46][1] - landmarks[44][1]

# get the width of the right eye 
def right_eye_width(landmarks):
    return landmarks[39][0] - landmarks[36][0]

# get the width of the left eye 
def left_eye_width(landmarks):
    return landmarks[45][0] - landmarks[42][0]

# width of left eyebrow
def left_eyebrow_width(landmarks):
    return landmarks[21][0] - landmarks[17][0]

# width of right eyebrow
def right_eyebrow_width(landmarks):
    return landmarks[26][0] - landmarks[22][0]

# Takes the average distance between the right eyebrow and the top 
# of the right eye
def right_eyebrow_eye_distance(landmarks):
    return (landmarks[37][1] - landmarks[19][1] + landmarks[38][1] - landmarks[20][1]) / 2

# Takes the average distance between the right eyebrow and the top 
# of the right eye
def left_eyebrow_eye_distance(landmarks):
    return (landmarks[43][1] - landmarks[23][1] + landmarks[44][1] - landmarks[24][1]) / 2

# Takes the distance between the bottom of the top lip and the top of the bottom lip
def mouth_gap(landmarks):
    return (landmarks[67][1] + landmarks[66][1] + landmarks[65][1] - landmarks[62][1] - landmarks[61][1] - landmarks[63][1])/3

#takes the average distance between the centers of the top of the top lip and the bottom of the bottom lip
def mouth_height_center_avg(landmarks):
    return (landmarks[58][1] + landmarks[57][1] + landmarks[56][1] - landmarks[50][1] - landmarks[51][1] - landmarks[52][1])/3

#takes the average distance between the left half of the top of the top lip and the bottom of the bottom lip
def mouth_height_left_avg(landmarks):
    return (landmarks[59][1] + landmarks[58][1] + landmarks[57][1] - landmarks[49][1] - landmarks[50][1] - landmarks[51][1])/3

#takes the average distance between the right half of the top of the top lip and the bottom of the bottom lip
def mouth_height_right_avg(landmarks):
    return (landmarks[57][1] + landmarks[56][1] + landmarks[55][1] - landmarks[51][1] - landmarks[52][1] - landmarks[53][1])/3

#Takes the height of the top of the top lip from y coordinate of the average of the corners of the mouth
def mouth_height_centerline_toplip(landmarks):
    return ((landmarks[48][1] + landmarks[54][1])/2 - landmarks[51][1])

#Takes the height of the bottom of the bottom lip from y coordinate of the average of the corners of the mouth
def mouth_height_centerline_bottomlip(landmarks):
    return (landmarks[57][1] - (landmarks[48][1] + landmarks[54][1])/2)

#Finds the width of the mouth by taking the difference between the corners of the mouth
def lip_width(landmarks):
    return landmarks[54][0] - landmarks[48][0]

# Finds the Y distance between the tip of the nose and the top of the top lip
def distance_between_nose_toplip(landmarks):
    return landmarks[51][1] - landmarks[33][1]

# Finds the distance between the outside corner of the left eye and the outside left corner of the mouth
def distance_between_left_eye_corner_left_lip_corner(landmarks):
    x = ((landmarks[48][0] - landmarks[36][0]), landmarks[48][1] - landmarks[36][1])
    return (x[0]**2 + x[1]**2)**(1/2)

# Finds the distance between the outside corner of the right eye and the outside right corner of the mouth
def distance_between_right_eye_corner_right_lip_corner(landmarks):
    x = ((landmarks[45][0] - landmarks[54][0]), landmarks[54][1] - landmarks[45][1])
    return (x[0]**2 + x[1]**2)**(1/2)

def generate_feature_vector(shape):
    # Convert Dlib object to form we can work with
    landmarks = shape_to_np(shape)
    # fill in feature vector
    data = np.zeros((18, 1))
    data[0] = right_eye_height(landmarks)
    data[1] = left_eye_height(landmarks)
    data[2] = right_eye_width(landmarks)
    data[3] = left_eye_width(landmarks)
    data[4] = left_eyebrow_width(landmarks)
    data[5] = right_eyebrow_width(landmarks)
    data[6] = right_eyebrow_eye_distance(landmarks)
    data[7] = left_eyebrow_eye_distance(landmarks)
    data[8] = mouth_gap(landmarks)
    data[9] = mouth_height_left_avg(landmarks)
    data[10] = mouth_height_center_avg(landmarks)
    data[11] = mouth_height_right_avg(landmarks)
    data[12] = mouth_height_centerline_toplip(landmarks)
    data[13] = mouth_height_centerline_bottomlip(landmarks)
    data[14] = lip_width(landmarks)
    data[15] = distance_between_nose_toplip(landmarks)
    data[16] = distance_between_left_eye_corner_left_lip_corner(landmarks)
    data[17] = distance_between_right_eye_corner_right_lip_corner(landmarks)
    
    # Normalize Data
    largest_val = -1
    for value in data:
        if value > largest_val:
            largest_val = value[0]
    for i in range(data.shape[0]):
        data[i] = data[i] / largest_val

    # Return
    return data
