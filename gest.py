import numpy as np
import cv2
import time





cap = cv2.VideoCapture(0)



# we stole this from the documentation
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0,255,(100,3))

def captureBG():
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    while True:
        ret,frame = cap.read()
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #frame = cv2.GaussianBlur(frame,(15,15),0)
        cv2.imshow('Capture background',frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            return frame

def captureGoodFeaturesToTrack(frame):
    return cv2.goodFeaturesToTrack(frame,mask=None,**feature_params)
    


    
def subtractBG(bg):
    ret,frame = cap.read()
    dkernel = np.ones((16,16), np.uint8)
    ekernel = np.ones((18,18),np.uint8)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#    cv2.imshow('1',frame)
    frame = cv2.absdiff(bg,frame)
#    cv2.imshow('2',frame)
    frame = cv2.GaussianBlur(frame,(15,15),0)
#    cv2.imshow('3',frame)
    ret,frame = cv2.threshold(frame, 6, 255, cv2.THRESH_BINARY )

#    cv2.imshow('4',frame)
    frame = cv2.erode(frame,ekernel,1)
#    cv2.imshow('5',frame)
    frame = cv2.dilate(frame,dkernel,2)
#    cv2.imshow('6',frame)
    return frame

color = np.random.randint(0,255,(100,3))


from subprocess import Popen, PIPE

rotate_left = '''keydown Control_L
keydown Shift_L
key R
keyup Control_L
keyup Shift_L
'''

rotate_right = '''keydown Control_L
key R
keyup Control_L
'''


go_left = '''keydown Left
keyup Left
'''
go_right = '''keydown Right
keyup Right
'''
def keypress(sequence):
    p = Popen(['xte'], stdin=PIPE)
    p.communicate(input=sequence)



def bgSubtractor():

    ret,f = cap.read()
    mask = np.zeros_like(f)
    bg = captureBG()
    ret,frame = cap.read()
    old_frame = None
    # set up optical flow
    while cap.isOpened:
        frame = subtractBG(bg)
        cv2.imshow('good features', frame)
        if cv2.waitKey(1) & 0xFF == ord('f'):
            old_frame = frame
            p0 = captureGoodFeaturesToTrack(frame)
            break
    i = 0

    state = "UNDEFINED"

    moving = False
    timer = 0
    ultimate_dx = 0
    ultimate_dy = 0
    frame_count = 0

    kk = 0
    while cap.isOpened:

        frame = subtractBG(bg)

        if cv2.waitKey(1) & 0xFF == ord('f'):
            old_frame = frame
            p0 = captureGoodFeaturesToTrack(frame)
            mask = np.zeros_like(mask)
        if kk % 20 == 0:
            old_frame = frame
            mask = np.zeros_like(mask)
            p0 = captureGoodFeaturesToTrack(frame)
        

        kk = kk + 1
        if p0 == None:
            continue
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame,frame, p0, None, **lk_params)


        if err != None:
            good_new = p1[st==1]
            good_old = p0[st==1]

        old_frame = frame.copy()
        frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)

        dx_tot = 0
        dy_tot = 0
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()

            dx_tot = dx_tot + (a-c)
            dy_tot = dy_tot + (b-d)
            cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            cv2.circle(frame,(a,b),5,color[i].tolist(),-1)

        cnt = min(len(good_new),len(good_old))

        state = "UNDEFINED"
        if cnt != 0:
            dy = dy_tot / cnt
            dx = dx_tot / cnt

            if time.time() - timer < 0.5:
                ultimate_dx += dx
                ultimate_dy += dy
                frame_count += 1
            else:
                if moving:
                    # we just went full DX. Never go full DX
                    if frame_count != 0:
                        final_dx = ultimate_dx / frame_count
                        final_dy = ultimate_dy / frame_count
                        # TODO detect move
                        if final_dx > 30:
                            print "LEFT"
                            keypress(go_left)
                        elif final_dx < -30:
                            print "RIGHT"
                            keypress(go_right)
                        elif final_dy > 20:
                            print "ROTATE_LEFT"
                            keypress(rotate_left)
                        elif final_dy < -20:
                            print "ROTATE_RIGHT"
                            keypress(rotate_right)

                        print "final_dx:%f" % final_dx
                        print "final_dy:%f" % final_dy
                    moving = False

                if abs(dx) > 30 or abs(dy) > 30:

                    print "dx:%f" % dx
                    print "dy:%f" % dy
                    timer = time.time()
                    ultimate_dx = 0
                    ultimate_dy = 0
                    frame_count = 0
                    moving = True



                




     
            
          

                # start timer

        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        p0 = good_new.reshape(-1,1,2)
            
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

bgSubtractor()
