'''
import cv2
import numpy
 
cap = cv2.VideoCapture(0)#调整参数实现读取视频或调用摄像头
while 1:
    ret, frame = cap.read()
    cv2.imshow("cap", frame)
    if cv2.waitKey(100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''

import cv2 as cv

image = cv.imread("D:/vcprojects/images/three.png")
capture = cv.VideoCapture(0)
detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_alt.xml")
while True:
    ret, image = capture.read()
if ret is True:
    cv.imshow("frame", frame)
    faces = detector.detectMultiScale(image, scaleFactor=1.05, minNeighbors=1,
    minSize=(30, 30), maxSize=(120, 120))
    for x, y, width, height in faces:
        cv.rectangle(image, (x, y), (x+width, y+height), (0, 0, 255), 2, cv.LINE_8, 0)
        cv.imshow("faces", image)
        c = cv.waitKey(50)
        if c == 27:
            break
        else:
            break

cv.destroyAllWindows()
