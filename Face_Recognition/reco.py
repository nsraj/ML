import cv2
import streamlit as st
import face_recognition as fr
import pickle
import os
import numpy as np
def realTimeFeed(_bool,isBoundingBox,isCap,name):
	stframe =st.empty()
	cap =cv2.VideoCapture(0,cv2.CAP_DSHOW)

	while _bool:
		_,frame = cap.read()

		if isBoundingBox:
			faceLocation = fr.face_locations(frame,model='hog')
			faceEmbeddings =fr.face_encodings(frame,faceLocation)
			for (top,right,bottom,left),faceEmbedding in zip(faceLocation,faceEmbeddings):
				cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
				if trainedModel:
					match =fr.compare_faces(trainmap["embedding"],faceEmbedding)
					faceDis = fr.face_distance(trainmap["embedding"],faceEmbedding)
					minFaceDis = min(faceDis)
					minFaceDisIdx = np.argmin(faceDis)

					if match[minFaceDisIdx] and minFaceDis < 0.5:
						label = trainmap['label'][minFaceDisIdx]
						cv2.putText(frame,f'{label}',(left-6,top-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

					else:
						cv2.putText(frame,'Unkown!',(left-6,top-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

				else:
					cv2.putText(frame,'Unkown!',(left-6,top-6),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
		stframe.image(frame,channels='BGR')
		if isCap:
			cv2.imwrite(f"{name}.jpg",frame)
			break

def train(name):
	faceimg = fr.load_image_file(f'{name}.jpg')
	facebb = fr.face_locations(faceimg)
	faceembd = fr.face_encodings(faceimg, facebb)
	for embd in faceembd:
		trainmap["embedding"].append(embd)
		trainmap["label"].append(name)
	st.write(trainmap)
	pickle.dump(trainmap, open("trainmodel.pkl","wb"))


st.title('Real Time Face Recognition')
tabs = ["Real Time: Feed", "Training"]
choice = st.sidebar.selectbox("Mode!", tabs)
trainmap = {"embedding":[],"label":[]}
trainedModel = False

if os.path.isfile("trainmodel.pkl"):
	trainmap = pickle.load(open("trainmodel.pkl","rb"))
	trainedModel = True
else:
	trainedModel = False


isCap = False
if choice == tabs[0]:
	isCap = False
	if st.button('Exit'):
		realTimeFeed(False,True,isCap,"")
	if st.button('Start'):
		realTimeFeed(True,True,isCap,"")
else:

	name = st.text_input("Enter your Nmae")
	if name != "":
		if st.button("Capture"):
			isCap = True
		if st.button("Train"):
			train(name)

		realTimeFeed(True,False,isCap,name )