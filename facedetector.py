import cv2 
import numpy as np
import os
from persons import getNameFromValue

cam = cv2.VideoCapture(0)
cascadesPath = os.getcwd()+"/cascades/"
guide = os.getcwd() + "/recogniser/training.yml"
faceCascade = cv2.CascadeClassifier(cascadesPath + "front2.xml")
recogniser = cv2.face.createLBPHFaceRecognizer()
recogniser.load(guide)

id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

while(True):
	ret, img = cam.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	npgray = np.array(gray, 'uint8')
	faces = faceCascade.detectMultiScale(
		gray, 
		scaleFactor=1.2, 
		minNeighbors=1,
		minSize=(30,30)
	)

	for (x,y,w,h) in faces:
		cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
		id = recogniser.predict(gray[y:y+h, x:x+h])
		name = getNameFromValue(id)
		cv2.putText(img, name, (x, y-10), font, 1,(0,0,255),2)

	cv2.imshow('face', img)
	if(cv2.waitKey(1) == ord('q')):
		break

cam.release()
cv2.destroyAllWindows()


