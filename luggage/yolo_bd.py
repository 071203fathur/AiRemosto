import cv2
import numpy as np
import time
from datetime import datetime 
import requests
import json
classes = ["Person","besar","kecil","sedang"]




net = cv2.dnn.readNetFromONNX("luggage/bestn_fix.onnx")
def luggage(frame):
    height, width, channels = frame.shape
    img = frame
    # img = cv2.resize(img, (1366,768))
    # img = cv2.resize(img, (900,680))
    # img = cv2.resize(img, (600,672))
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net.setInput(blob)
    detections = net.forward()[0]
    

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    # x_scale = img_width/640
    # y_scale = img_height/640
    x_scale = 1
    y_scale = 1
    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence > 0.5: #confidence level -> accuracy
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] > 0.5:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx- w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
    fulldata = [0]*7
    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.3)


    for i in indices:
        x1,y1,w,h = boxes[i]
        label = classes[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img,(x1,y1),(x1+w,y1+h),(0,255,0),2)
        cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,127,255),2)

        # color = (255, 255,0)
        if label == "besar":
            color = (255,0,0)
            cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color,2)
        if label == "sedang":
            color = (255,0,255)
            cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color,2)
        if label == "kecil":
            color = (0,255,255)
            cv2.putText(img, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
            cv2.rectangle(img,(x1,y1),(x1+w,y1+h),color,2)
    
    return img
