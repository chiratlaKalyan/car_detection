import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*

#Assign pre-trained weights
#model=YOLO('runs/detect/yolov8n_cars_custom/weights/best.pt')
model=YOLO('yolov8n.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('BR Vehicle Tracking')
cv2.setMouseCallback('BR Vehicle Tracking', RGB)

cap=cv2.VideoCapture('images/video.mp4')
#cap=cv2.VideoCapture(0)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 


tracker=Tracker()
count=0
car_count=1
vh_down={}
counter=[]
cy1=203
cy2=303
offset=10

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    list=[]
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            list.append([x1,y1,x2,y2])
            cv2.putText(frame, str(c), (x1,y1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)

    bbox_id=tracker.update(list)

    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        #print('TERROR',str(cx),str(cy))
        if cy1 < (cy+offset) and cy1 > (cy-offset):
           cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
           cv2.putText(frame,str(car_count),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
           cv2.
           car_count+=1


    #Border1
    cv2.line(frame,(179,cy1),(850,cy1),(255,255,255),1)
    cv2.putText(frame, "Border1", (178, cy1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (243, 250, 18), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

