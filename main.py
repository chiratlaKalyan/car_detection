import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
from readNumPlate import*


"""My Notes
#CCTV Camera Resolution has to be reduced to desired frame size i.e 1280*720
 
"""

#Assign pre-trained weights
model=YOLO('yolov8n_best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :
        colorsBGR = [x, y]
        print(colorsBGR)


cv2.namedWindow('BR Vehicle Tracking')
cv2.setMouseCallback('BR Vehicle Tracking', RGB)

cap=cv2.VideoCapture('images/veh2.mp4')
#cap=cv2.VideoCapture(0)

my_file = open("object_classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Tracker Class & Image Reading Class initialization.
tracker=Tracker()
readImage=imgOCR()

count=0
car_count=1
vh_detected={}
detected_cars={}

video_res_width=1020
video_res_height=600
offset=6
offset2=video_res_height//3    #eg 200


#To draw Lines Eg: Spot1,Spot2
cy1=video_res_height-offset2  #eg 400
cx1=offset*8  #eg 100
cx2=video_res_width-cx1 #920
cy2=cy1+100
numberPlates={}


while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(video_res_width,video_res_height))


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

    bbox_id=tracker.update(list)

    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2

        if cy1 < (cy+offset) and cy1 > (cy-offset) and id not in vh_detected:
               vh_detected[id]=cy
        #if id not in vh_detected:
               #del(vh_detected[id])
               cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
               cv2.putText(frame, str(car_count), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
               #saving image
               image_name='images/detected_car'+str(car_count)+'.jpg'
               if not cv2.imwrite(image_name,frame):
                      raise Exception("Could not write image")
               else:
                      print(image_name)
                      #readImage.readTextFromNumPlate(image_name)
                      #Num_plate_text=imgOCR.readTextFromNumPlate(image_name)
                      #numberPlates[car_count]=Num_plate_text
                      #print(Num_plate_text)
               car_count += 1

    cv2.line(frame, (cx1, cy1), (cx2, cy1), (255, 255, 255), 1)
    cv2.putText(frame, "Detecting Point", (cx1-offset, cy1), cv2.FONT_HERSHEY_COMPLEX, 0.8, (243, 250, 18), 1)

    cv2.imshow("BR Vehicle Tracking", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
