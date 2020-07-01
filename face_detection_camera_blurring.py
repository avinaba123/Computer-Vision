import cv2
import matplotlib.pyplot as plt
import numpy as np

fd = cv2.CascadeClassifier('E:/original/Computer-Vision-with-Python/DATA/haarcascades/haarcascade_frontalface_default.xml')

def detect_and_blur_face(img):
    im_cp = img.copy()
    rnp_rect = fd.detectMultiScale(im_cp,scaleFactor=1.2,minNeighbors=5)
    
    for x,y,h,w in rnp_rect:
        roi = im_cp[y:y+w,x:x+h]
        bi = cv2.medianBlur(roi,55)
        im_cp[y:y+w,x:x+h] = bi
    return im_cp

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    frame = detect_and_blur_face(frame)
    cv2.imshow('frame',frame)
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()