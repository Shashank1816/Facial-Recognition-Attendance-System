import cv2
import numpy as np
import face_recognition
#we create a list that will create a list of all the images in the folder ImagesBasic and create it's encodings
import os
from datetime import datetime

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
#We'll create the function to mark our attendance
def markAttendance(name):
    with open('Attendance.csv','r+') as f: #to open the file and we want to read and write at the same time
        #now we'll read in all the lines currently in our data, so that somebody arrives twice we don't want to repeat the entry
        myDataList=f.readlines()
        nameList=[]#we want to put all the names already present in this list
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        #once we have this list complete, we'll check that the current name is already present in the list or not
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S') #format for time
            f.writelines(f'\n{name},{dtString}')


        print(myDataList)


encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print("Encoding Complete")

# The third step is to find the matches between our encodings. the images to be matched is gonna come from our webcam

cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)#We want to reduce the size of the image to speed up the process the scale is 0.25 i.e one-fourth of the original size
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)# We'll convert the image to rgb

    #Now we can find multiple images in our webcam so for that we need to find the locations of our images and then we'll send in the locations of our images to the encoding function
    #to find the location
    facesCurFrame= face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)
    #we'll iterate through all the faces in our current frame
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame): #one by one it'll grab one face location and it's encoding
        #then we'll perform the matching
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        #then we'll find the distance
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)#The lowest distance will be our best match
        #print(faceDis)
        matchIndex=np.argmin(faceDis) # to get the image with the lowest distance

        if matches[matchIndex]:
            name = classNames[matchIndex].upper() #to convert uppercase
            #print(name)
            #we'll create a rectangle around the face
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    #We can create a bounding box around the face and then we'll print the name
    cv2.imshow('Webcam',img)
    cv2.waitKey(1)