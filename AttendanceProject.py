import cv2
import numpy as np
import face_recognition
#we create a list that will create a list of all the images in the folder ImagesBasic and create it's encodings
import os

path = 'ImagesAttendance'
images = []
classNames =[]
mylist=os.listdir(path)
print(mylist)
#next we'll use these names and import these images one by one
for cl in mylist:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])#so that .jpg doesn't come in the names

print(classNames)
#next we start with our encoding process, we'll find the encodings for each and every image
def findEncodings(images):
    encodeList=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)#convert the images to rgb
        encode =face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print("Encoding Complete")

# The third step is to find the matches between our encodings. the images to be matched is gonna come from our webcam

cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)#We want to reduce the size of the image to speed up the process the scale is 0.25 i.e one-fourth of the original size
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)# We'll convert the image to rgb
    #Now we can find multiple images in our 

