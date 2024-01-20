import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('people.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()

cy1=200
cy2=220
offset=6
p_down={}
p_up={} 
counter_down=[]
counter_up=[]
while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
   
    results=model.predict(frame)
    boxes=results[0].boxes.data.tolist()
    list=[]
    for box in boxes:
        x1,y1,x2,y2,score,class_id=box
        x1,y1,x2,y2,score,class_id= int(x1),int(y1),int(x2),int(y2),int(score),int(class_id)
        if 'person' in class_list[int(class_id)]:
            list.append([int(x1),int(y1),int(x2),int(y2)])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        cv2.rectangle(frame, (x3,y3),(x4,y4),(220,2230,2250), 2)
        cv2.putText(frame,str(id),(x3-2,y3-3),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        # For going Down
        if cy1<(cy+offset) and cy1>(cy-offset):
            p_down[id]=cy
        if id in p_down:
            if cy2<(cy+offset) and cy2>(cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                if counter_down.count(id)==0:
                    counter_down.append(id)



        # For Going Up
        
        if cy2<(cy+offset) and cy2>(cy-offset):
            p_up[id]=cy
        if id in p_up:
            if cy1<(cy+offset) and cy1>(cy-offset):
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                if counter_up.count(id)==0:
                    counter_up.append(id)
           


    cv2.line(frame,(0,cy1),(1018,cy1),(255,255,255),1)
    #cv2.putText(frame,'First Line',(110,cy1-6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(220,255,0),2)


    cv2.line(frame,(0,cy2),(1018,cy2),(255,255,255),1)
    #cv2.putText(frame,"Second Line",(110,cy2-6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(220,255,0),2)
    d=len(counter_down)
    cv2.putText(frame,f"Going_down:-{d}",(60,80),cv2.FONT_HERSHEY_SIMPLEX,0.8,(220,255,0),2)
    
    u=len(counter_up)
    cv2.putText(frame,f"Going_Up:-{u}",(60,40),cv2.FONT_HERSHEY_COMPLEX,0.8,(220,255,0),2)
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0)&0xFF==ord('d'):
        break
cap.release()
cv2.destroyAllWindows()
