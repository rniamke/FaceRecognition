import numpy as np
import cv2
import os
import sys
from PIL import Image
import time
import json

from PyQt5.QtWidgets import (QInputDialog, QApplication)
app = QApplication(sys.argv)



# For face detection we will use the Haar Cascade provided by OpenCV.
#cascadePath = "haarcascade_frontalface_default.xml"
#faceCascade = cv2.CascadeClassifier(cascadePath)

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)

# images will contains face images
images = []
# labels will contains the label that is assigned to the image
labels = []

faces_dict = {} # lbl:name

faces_db = os.path.expanduser('~/FaceRecognition/opencv_faces_db/')

hasTrained = False


# For face recognition we will use the LBPH Face Recognizer 
recognizer = cv2.face.createLBPHFaceRecognizer()
#recognizer = cv2.face.createFisherFaceRecognizer()
#recognizer = cv2.face.createEigenFaceRecognizer()


# First, Read the faces database dictionary :
try:
    with open(os.path.join(faces_db, 'faces_dict.json')) as fp:
        faces_dict = json.load(fp)
    #print (faces_dict)
except:
    pass

# Then, read images ans set theis labels from dictionary
for root, dirs, files in os.walk(faces_db):
    for dirname in dirs:
        dirpath = os.path.join(root, dirname)
        lbl = int(faces_dict[dirname])
        for fname in os.listdir(dirpath):
            #print("fname={}".format(fname))
            image_path = (os.path.join(dirpath, fname))
            image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            faces = faceCascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                #images.append(image[y: y + h, x: x + w])
                images.append(image)
                labels.append(lbl)
                #cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                #cv2.waitKey(50)
    if len(images) > 1:
        recognizer.train(images, np.array(labels))
        hasTrained = True


video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CV_FEATURE_PARAMS_HAAR
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        name = 'Inconnu'
        # Predit face :
        if hasTrained:
            #print('Predicting...')
            gray_image = cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
            nbr_predicted = recognizer.predict(gray_image)
            #print (faces_dict)
            #print('key predicted: {}'.format(nbr_predicted))
            name = faces_dict[str(nbr_predicted)]
        # Write some Text
        cv2.putText(frame, name, (x-10,y), font, 1, (255,255,255), 2)


    # Display the resulting frame
    cv2.imshow('Video', frame)


    keyPressed = cv2.waitKey(25) & 0xFF


    if keyPressed == ord('a'):
        # Append face and label to array
        for (x, y, w, h) in faces:
            gray_image = cv2.cvtColor(frame[y: y + h, x: x + w], cv2.COLOR_BGR2GRAY)
            cv2.imshow('face', gray_image)
            text, ok = QInputDialog.getText(QInputDialog(), 'Input Dialog', 'Entrer une étiquette:')
            cv2.destroyWindow('face')
            if ok:
                num = 0
                nom = str(text)
                if nom in faces_dict:
                    num = int(faces_dict[nom])
                else:
                    num = len(faces_dict)
                faces_dict[str(num)] = nom
                faces_dict[nom] = str(num)
                # write face on disk
                face_dir = os.path.join(faces_db, nom)
                #print (face_dir)
                try:
                    os.makedirs(face_dir)
                except:
                    pass
                fnth = len(os.listdir(face_dir))
                fname = os.path.join(face_dir, nom + str(fnth) + '.png')
                #print(fname)
                cv2.imwrite(fname, gray_image)
                images.append(gray_image)
                labels.append(num)
                with open(os.path.join(faces_db, 'faces_dict.json'), 'w') as fp:
                    json.dump(faces_dict, fp, indent=4, separators=(',',':'))
                print('Face added.')
        # Perform the tranining
        if len(images) > 1:
            recognizer.train(images, np.array(labels))
            hasTrained = True
            print('Trained.')
    elif keyPressed == ord('p'):
        for i, image in enumerate(images):
            cv2.imshow('face', image)
            #cv2.imwrite('opencv_face'+str(i)+'.png', image)
            #print (image)
            time.sleep(2)
            cv2.destroyWindow('face')
    elif keyPressed == ord('q'):
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
#--- END ---



