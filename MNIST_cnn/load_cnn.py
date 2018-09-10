import tensorflow
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("model.h5")
print("Loaded model from disk")
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

cap = cv2.VideoCapture(0)

while(1):
	ret, frame = cap.read()
	cv2.imshow('Frame', frame)
	k = cv2.waitKey(10)
	
	if k == ord('a'):
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		blur = cv2.GaussianBlur(gray, (5, 5), 0)
		v = np.median(blur)
		edge = cv2.Canny(blur, 0.66 * v, 1.33 * v)
		cv2.imshow('Edge', edge)
		kernel = np.ones((7,7),np.uint8)
		dilation = cv2.dilate(edge, kernel, iterations = 1)
		_, contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		rects = [cv2.boundingRect(cnt) for cnt in contours]

		for rect in rects:
			size = int(rect[3] * 1.1)
			
			pt1 = int(rect[1] + (rect[3] // 2) - (size // 2))
			pt2 = int(rect[0] + (rect[2] // 2) - (size // 2))
			
			if pt1 <=0 or pt2 <= 0 or size < 10 or rect[2] < 10:
				continue
			
			roi = dilation[pt1: pt1 + size, pt2: pt2 + size]
			roi = 255 - roi
			roi = cv2.resize(roi, (28, 28), cv2.INTER_CUBIC)
			cv2.imshow('RoI', roi)
			cv2.waitKey(0)
			n, lar = 0, 0
			roi = roi / 255
			roi = 1 - roi
			roi = np.array([roi.reshape((28, 28, 1))])
			num = classifier.predict(roi)
			print(num[0])
			for i, j in enumerate(num[0]):
				if j > lar:
					n = i
					lar = j
			cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
			cv2.putText(frame, str(n), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
		
		cv2.imshow('Dilation', dilation)
		cv2.imshow('Prediction', frame)
		cv2.waitKey(0)
	
	cv2.destroyWindow('Dilation')
	cv2.destroyWindow('Prediction')
	cv2.destroyWindow('Edge')
	cv2.destroyWindow('RoI')
	if k == 27:
		break

cv2.destroyAllWindows()
cap.release()
