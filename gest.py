import numpy as np
import cv2

"""cap = cv2.VideoCapture(1)
fgbg = cv2.BackgroundSubtractorMOG(10, 10, 0.8,)

def captureBackground():
    ret,frame = cap.read()
    while True:
        ret,frame = cap.read()
        cv2.imshow('lol', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            fgbg.apply(frame)
            return

captureBackground()
while(cap.isOpened):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    fgmask = fgbg.apply(frame)

    # Display the resulting frame
    cv2.imshow('frame',fgmask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
a
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
"""

cap = cv2.VideoCapture(0)
def captureBG():
    ret,frame = cap.read()

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame,(5,5),0)
        cv2.imshow('Capture background',frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            return frame

def bgSubtractor():
    bg = captureBG()
    dkernel = np.ones((15,15), np.uint8)
    ekernel = np.ones((15,15),np.uint8)
    while cap.isOpened:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.absdiff(bg,frame)
        cv2.imshow('jemoeder',frame)
        ret,frame = cv2.threshold(frame, 30, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('frame', frame)
        frame = cv2.erode(frame,ekernel,3)
        cv2.imshow('frameErode', frame)
        frame = cv2.dilate(frame,dkernel,3)
        cv2.imshow('frameDilate', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

bgSubtractor()
