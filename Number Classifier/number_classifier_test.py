import numpy as np
from sklearn.externals import joblib
import cv2
from skimage.feature import hog

classifier = joblib.load('num_classifier.pkl')

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
		kernel = np.ones((7, 7), 'uint8')
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
			roi = cv2.GaussianBlur(roi, (3, 3), 0)
			roi = cv2.resize(roi, (28, 28), cv2.INTER_CUBIC)
			cv2.imshow('RoI', roi)
			if cv2.waitKey(0) == ord('q'):
				break
			n, lar = 0, 0
			roi = 255 - roi
			
			hog_feature = hog(roi, orientations = 9, pixels_per_cell = (14, 14), cells_per_block = (1, 1), visualise = False)
			num = classifier.predict(np.array([hog_feature]))
			prob = classifier.predict_proba(np.array([hog_feature]))
			print(prob, num)
			cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
			cv2.putText(frame, str(int(num[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2)
		
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
