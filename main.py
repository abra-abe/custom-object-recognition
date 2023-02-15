import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import cv2 as cv #Please install with PIP: pip install cv2

TM_DATA = None
model = None
cap = None
ret = None
frame = None
PredictionVariable = None
key = None


print('START')
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
TM_DATA = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def imgDetection():
    img = cv.imread('bottle1.jpg')

    while True:
        img = cv.resize(img, (224, 224))
        image_array = np.asarray(img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        TM_DATA[0] = normalized_image_array
        PredictionVariable = model.predict(TM_DATA)
        print('Prediction:')
        PredictionVariable = PredictionVariable * 100
        print(PredictionVariable)

        #Converting to one dimensional array
        PredictionVariable = PredictionVariable.ravel()

        if ((PredictionVariable[0] > PredictionVariable[1]) and (PredictionVariable[0] > 50)):
            print('fish')
        elif ((PredictionVariable[1] > PredictionVariable[0]) and (PredictionVariable[1] > 50)):
            print('plastic')
        else:
            print('no object detected')

        cv.imshow('output', img)
        key = cv.waitKey(0)
        if key == (ord('q')):
            break
        cv.destroyAllWindows()
        img.release()

#
# def vidDetection():
#     cap = cv.VideoCapture("https://192.168.43.1:8080/video")
#
#     cap.set(3, 640)
#     cap.set(4, 480)
#
#     while True:
#         ret , frame = cap.read()
#         cv.imshow('Window',frame)
#         frame = cv.resize(frame, (224, 224))
#         image_array = np.asarray(frame)
#         # Normalize the image
#         normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
#         # Load the image into the array
#         TM_DATA[0] = normalized_image_array
#         PredictionVariable = model.predict(TM_DATA)
#         print('Prediction:')
#         PredictionVariable = PredictionVariable * 100
#         print(PredictionVariable)
#
#         #Converting to one dimensional array
#         PredictionVariable = PredictionVariable.ravel()
#
#         if ((PredictionVariable[0] > PredictionVariable[1]) and (PredictionVariable[0] > 50)):
#             print('fish')
#         elif ((PredictionVariable[1] > PredictionVariable[0]) and (PredictionVariable[1] > 50)):
#             print('plastic')
#         else:
#             print('no object detected')
#
#         key = cv.waitKey(2000)
#         if key == (ord('q')):
#             break
#     cv.destroyAllWindows()
#     cap.release()


def vidDetection():
    cap = cv.VideoCapture(0)
    while True:
        ret , frame = cap.read()
        cv.imshow('Window',frame)
        frame = cv.resize(frame, (224, 224))
        image_array = np.asarray(frame)
        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # Load the image into the array
        TM_DATA[0] = normalized_image_array
        PredictionVariable = model.predict(TM_DATA)
        print('Prediction:')
        PredictionVariable = PredictionVariable * 100
        print(PredictionVariable)

        #Converting to one dimensional array
        PredictionVariable = PredictionVariable.ravel()

        if ((PredictionVariable[0] > PredictionVariable[1]) and (PredictionVariable[0] > 50)):
            print('fish')
        elif ((PredictionVariable[1] > PredictionVariable[0]) and (PredictionVariable[1] > 50)):
            print('plastic')
        else:
            print('no object detected')

        key = cv.waitKey(2000)
        if key == (ord('q')):
            break
    cv.destroyAllWindows()
    cap.release()


option = input("1 or 2")
print("\n")
if option == '1':
    imgDetection()
elif option == '2':
    vidDetection()
else:
    print('invalid option')

print('TNE END')
