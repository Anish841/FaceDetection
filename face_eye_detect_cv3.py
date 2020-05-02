import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
    #flags = cv2.CV_HAAR_SCALE_IMAGE
)

for (x,y,w,h) in faces:
	# To draw a rectangle in a face 
	cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),2) 
	roi_gray = gray[y:y+h, x:x+w] 
	roi_color = image[y:y+h, x:x+w]
	# Detects eyes of different sizes in the input image 
	eyes = eye_cascade.detectMultiScale(roi_gray) 
	#To draw a rectangle in eyes 
	for (ex,ey,ew,eh) in eyes: 
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)

	# Display an image in a window 
	cv2.imshow('image',image) 
	cv2.imwrite("faceEye.png",image)
cv2.waitKey(0)
