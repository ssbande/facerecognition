import os 
import cv2 
from PIL import Image 
import numpy as np
from persons import people

recogniser = cv2.face.createLBPHFaceRecognizer()
dataPath = os.getcwd()+"/data/"


try:
	if not os.path.exists("recogniser/"):
	    os.makedirs("recogniser/")
except OSError:
	print(OSError)

def getImagesWithUser():
	persons = [x[0] for x in os.walk(dataPath)][1:]
	print(persons)
	faces = []
	ids = []

	for person in persons:
		name = person.split('/')[-1]
		# personId = personId + 1
		try:
			personId = people[name]
		except:
			personId = people['unknown']
		print("person: " + name)
		print("personId: " + str(personId))
		imagePaths = [os.path.join(dataPath, name, f) for f in os.listdir(person)]
		print(imagePaths)

		for imagePath in imagePaths:
			faceImg = Image.open(imagePath).convert('L')
			face = np.array(faceImg, 'uint8')
			ids.append(personId)
			faces.append(face)
			cv2.imshow('training', face)
			cv2.waitKey(10)

	return np.array(ids), faces

Ids, faces = getImagesWithUser()
print(Ids)
print(faces)
recogniser.train(faces, Ids)
recogniser.save('recogniser/training.yml')
cv2.destroyAllWindows()
