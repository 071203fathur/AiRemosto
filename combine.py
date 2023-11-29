import cv2, time, numpy as np
from age.main import age

gender_model = cv2.dnn.readNetFromONNX("gender/gender.onnx") 
net_luggage = cv2.dnn.readNetFromONNX("luggage/bestn_fix.onnx")
classes_gender = ['pria', 'wanita']
cap = cv2.VideoCapture(1)

def gender(cap, img_d):
    _, img = cap.read()
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    gender_model.setInput(blob)
    detections = gender_model.forward()[0]

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence >= 0.25:
            classes_score = row[5:]
            ind = np.argmax(classes_score)
            if classes_score[ind] >= 0.25:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx-w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
             

    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.25,0.25)
    
    g_detect=[0,0]
    for j in indices:
        x1,y1,w,h = boxes[j]
        label = classes_gender[classes_ids[j]]
        if label == "Pria":
            g_detect[0] += 1
        if label == "Wanita":
            g_detect[1] += 1
        color = (0, 225, 00)
        cv2.rectangle(img_d,(x1,y1),(x1+w,y1+h),(150,200,0),2)
        cv2.putText(img_d, label, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.5,(150,255,0),2)
  
    return img_d


classes_luggage  = ["Person","besar","kecil","sedang"]

def luggage(cap, img_d):
    _, img = cap.read()
    height, width, channels = img.shape
    barang = []
    blob = cv2.dnn.blobFromImage(img,scalefactor= 1/255,size=(640,640),mean=[0,0,0],swapRB= True, crop= False)
    net_luggage.setInput(blob)
    detections = net_luggage.forward()[0]

    

    # cx,cy , w,h, confidence, 80 class_scores
    # class_ids, confidences, boxes

    classes_ids = []
    confidences = []
    boxes = []
    rows = detections.shape[0]

    img_width, img_height = img.shape[1], img.shape[0]
    x_scale = img_width/640
    y_scale = img_height/640

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
        label = classes_luggage[classes_ids[i]]
        conf = confidences[i]
        text = label + "{:.2f}".format(conf)
        cv2.rectangle(img_d,(x1,y1),(x1+w,y1+h),(0,255,0),2)
        cv2.putText(img_d, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,(0,127,255),2)

        # color = (255, 255,0)
        if label == "besar":
            color = (255,0,0)
            cv2.putText(img_d, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
            cv2.rectangle(img_d,(x1,y1),(x1+w,y1+h),color,2)
        if label == "sedang":
            color = (255,0,255)
            cv2.putText(img_d, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
            cv2.rectangle(img_d,(x1,y1),(x1+w,y1+h),color,2)
        if label == "kecil":
            color = (0,255,255)
            cv2.putText(img_d, text, (x1,y1-2),cv2.FONT_HERSHEY_COMPLEX, 0.7,color,2)
            cv2.rectangle(img_d,(x1,y1),(x1+w,y1+h),color,2)
        
        if label != "Person":
            barang.append(label)
    return img_d

classes_race = ['negroid','east_asian','indian','latin','middle_eastern','south_east_asian','kaukasia']
net = cv2.dnn.readNetFromONNX("race/best.onnx")

# while True:
def race(cap, img_d):
    _, img = cap.read()
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
    x_scale = img_width/640
    y_scale = img_height/640

    for i in range(rows):
        row = detections[i]
        confidence = row[4]
        if confidence >= 0.3:
            classes_score = row
            ind = np.argmax(classes_race)
            if classes_score[ind] >= 0.3:
                classes_ids.append(ind)
                confidences.append(confidence)
                cx, cy, w, h = row[:4]
                x1 = int((cx-w/2)*x_scale)
                y1 = int((cy-h/2)*y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)
                box = np.array([x1,y1,width,height])
                boxes.append(box)
                

    fulldata = [0]*7
    indices = cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.3)
    #print(indices)

    for j in indices:
        n_classes = len(classes_race)-1
        if j > n_classes:
            j = n_classes 
        label = "ras terdeteksi : " + classes_race[j]
        color_text = (0, 150,255)
        fulldata[j] =+ 1
        cv2.putText(img_d, label, (20,150),cv2.FONT_HERSHEY_COMPLEX, 0.5,color_text,2)
    return img_d

while True:
    img = age(cap)
    img = luggage(cap, img)
    img = race(cap, img)
    img = gender(cap, img)
    cv2.imshow("REMOSTO COMBINE ENGINE!!!",img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
