#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

import numpy as np
import math
import cv2
import time
from matplotlib import pyplot as plt
#from matplotlib import pyplot as plt
#from picamera.array import PiRGBArray
#from picamera import PiCamera

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 20,
                       qualityLevel = 0.3,
                       minDistance =7,
                       blockSize = 7 )

class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture("C:/Year4/EEM449 Final Year Project/Videos/experiment51.h264") #PiCamera()
        #self.cap = cv2.VideoCapture(0)         #for PiCamera module or webcam
        #self.cam.set(3,1280)                   #for webcam
        #self.cam.set(4,720)                    #for webcam
        #self.cam.resolution = (640, 480)       #for PiCamera module
        #self.cam.framerate = 32                #for PiCamera module or webcam
        #self.rawCapture = PiRGBArray(self.cam, size=(640, 480))    #for PiCamera module
        self.frame_idx = 0
        # allow the camera to warmup
        time.sleep(0.1)
        # initialize variables
        self.max_points=50
        self.first_pic=0
        self.frame_count=0
        self.absolute_x=0
        self.absolute_y=0
        self.absolute_xx=0
        self.absolute_yy=0
        self.carWid=0.151
        self.pipeDia=0.3
        self.camOffset=0.06#54
        self.VFOV=math.radians(37.16)
        self.HFOV=math.radians(53.50)
        self.inclination=math.radians(90-40)
        #Calcluations of constants for 1st method
        self.verticalDis=self.pipeDia/2 + math.sqrt(4*(self.pipeDia/2)**2-self.carWid**2)/2 - self.camOffset
        self.resolution=(2*self.verticalDis*math.tan(self.VFOV/2))/720


    #Function for 2nd method (video rotated for 0 or 180 degree)
    def convert_y(self,point,height):
        beta=math.atan(2*(point/height)*math.tan(self.VFOV/2))      #finding angle beta
        y=self.verticalDis/math.tan(self.inclination+beta)          #retrieving real world distance
        return y
            
    def run(self):
        ############################# Feature tracking starts here ########################
        while 1:
        #for frame in self.cam.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text

            (grabbed, image) = self.cam.read()
            if not grabbed:
                break
            #image = frame.array                #for PiCamera module
            self.frame_count+=self.frame_count
            height, width,channel = image.shape
            size=180
            frame_crop = image[(height/2):(height/2+size*2),(width/2-size):(width/2+size)]
            if self.first_pic==0:
                #cv2.line(frame_crop,(size,0),(size,size*2),(255,0,0),2)
                cv2.line(image,(0,height/2),(width,height/2),(255,0,0),2)
                #cv2.imshow('First', frame_crop)
                print 'The vertical distance of camera relative to pipe surface is {:.4f}m'.format(self.verticalDis)
                cv2.imshow('First', image)
                self.first_pic=1
            frame_gray = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2GRAY)    
            vis = frame_crop.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good): #loop for tracking for each predetermined point
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                cv2.putText(vis, 'track count: %d' % len(self.tracks),(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            if self.frame_idx % self.detect_interval == 0: #checking old points and track for new points if necessary
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                if len(self.tracks)>=self.max_points:
                    continue
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            #cv2.imshow('lk_track', vis)
            
            ###################### measurement of angle and distance starts here ####################
            
            deg=np.zeros(1)
            delta_x=np.zeros(1)
            delta_y=np.zeros(1)
            #converted_x=np.zeros(1)
            converted_y=np.zeros(1)
            for tracks in range (0,(len(self.tracks)-1)):
                test=np.squeeze(np.asarray(self.tracks[tracks:tracks+1]))   #make one of the track point as an array, and reduce its dimension
                if len(test)<=2:                                            #check to prevent error,check not to use new track point
                    continue
                
                delta_x=np.append(delta_x,(test[-1,0]-test[-2,0]))
                delta_y=np.append(delta_y,(test[-1,1]-test[-2,1]))

                ########### Method 2 by retriving the true distance of the point ###################
                
                y1=self.convert_y(test[-1,1],height)
                y2=self.convert_y(test[-2,1],height)
                converted_y=np.append(converted_y,y2-y1)
                
            self.absolute_y+=np.average(delta_y)*self.resolution    #Averaging and real distance conversion for Method 1
            self.absolute_yy+=np.average(converted_y)               #Averaging for Method 2
            cv2.putText(vis, 'Distance y: %.4f m' % self.absolute_y,(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            cv2.putText(vis, 'Distance yy: %.4f m' % self.absolute_yy,(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            #plt.plot(self.frame_count,self.absolute_y)
            
            print self.absolute_y, self.absolute_yy
            #print delta_y
            #print converted_y
            cv2.imshow('lk_track', vis)
            cv2.line(image,(0,height/2),(width,height/2),(255,0,0),2)
            cv2.imshow('Now', image)
            
            # clear the stream in preparation for the next frame
            #self.rawCapture.truncate(0)'''         #for PiCamera module

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27 or ch ==ord("q"):
                break
            if ch ==ord("r"):
                self.absolute_x=self.absolute_y=self.absolute_xx=0
                
def main():
    #import sys
    #try: video_src = sys.argv[1]
    #except: video_src = 0

    #print __doc__
    #App().distance()
    global skip_count
    skip_count=0
    App().run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
