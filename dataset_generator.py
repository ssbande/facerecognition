import cv2 
import os

cam = cv2.VideoCapture(0)
cascadesPath = os.getcwd()+"/cascades/"
faceCascade = cv2.CascadeClassifier(cascadesPath + "front2.xml")

id = raw_input("For which person you want to train me ? ")
process = True

try:
	if not os.path.exists("data/"+id):
	    os.makedirs("data/" +id)
	else:
		print('already existing data - no sampling done')
		process = False
except OSError:
	print(OSError)

samples = 0
print('samples: ', str(samples))
while(process):
	ret, frame = cam.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = faceCascade.detectMultiScale(
		gray, 
		scaleFactor = 1.2,
		minNeighbors = 1,
		minSize = (30,30)
	)

	for(x, y, w, h) in faces:
		samples = samples + 1
		cv2.imwrite("data/{0}/{1}.jpg".format(id, str(samples)), gray[y:y+h, x:x+w])
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.waitKey(100)


	cv2.imshow('face', frame)
	cv2.waitKey(1)
	if (samples) > 49:
		print("sampling done for " + id)
		break

cam.release()
cv2.destroyAllWindows()
