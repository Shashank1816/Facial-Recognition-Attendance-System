import cv2
import numpy as np
import face_recognition
#print('All necessary libraries imported!')

#First step is loading the images and converting them into RGB, because we get the image as BGR but the library understands the image as RGB

#What we are gonna do is first we will test the image of barack obama and find the encodings and then we'll test our model with the test images whether it's of barack obama or not

#firstly, we'll need to import our image
imgBar=face_recognition.load_image_file('ImagesBasic/Barack_obama.jpg')# then we'll convert this to RGB, since it's BGR
imgBar=cv2.cvtColor(imgBar,cv2.COLOR_BGR2RGB)



#Similarly import our test image
imgTest=face_recognition.load_image_file('ImagesBasic/barack_obama_test.jpg')
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

# The next step is to find the faces in our images and then their encodings
faceLoc=face_recognition.face_locations(imgBar)[0]#[0] because we just uploaded a single image
#now we'll encode the face that we have detected
encodeBar = face_recognition.face_encodings(imgBar)[0]
#to see where we have detected the face
cv2.rectangle(imgBar,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)#these are the four corners of the rectangle

faceLocTest=face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)#these are the four corners of the rectangle

#the third and the final step is to compare these two faces and find the distance between the encodings
#in the backend we'll use Linear SVM to check whether they are equivalent or not

results = face_recognition.compare_faces([encodeBar],encodeTest)
faceDis=face_recognition.face_distance([encodeBar],encodeTest)
print(results)# if true that means it has found the similar encodings and false if the encodings are different
print(faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

#display the image
cv2.imshow('Barack Obama',imgBar) # this means the name is Barack Obama and we want to display imgBar image
cv2.imshow('Barack Obama test',imgTest)
cv2.waitKey(0)# this means 0 delay
