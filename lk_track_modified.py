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
#from picamera.array import PiRGBArray
#from picamera import PiCamera

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 20,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture("C:/Year4/EEM449 Final Year Project/Videos/test90v24.h264") #PiCamera()
        #self.cap = cv2.VideoCapture(0)
        #self.cam.set(3,1280)
        #self.cam.set(4,720)
        #self.cam.resolution = (640, 480)
        #self.cam.framerate = 32
        #self.rawCapture = PiRGBArray(self.cam, size=(640, 480))
        self.frame_idx = 0
        # allow the camera to warmup
        time.sleep(0.1)
        self.absolute_x=0
        self.previous_x=np.zeros((0,0))
            
    def run(self):
        while 1: #count!=0:
        #for frame in self.cam.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
	# grab the raw NumPy array representing the image, then initialize the timestamp
	# and occupied/unoccupied text
            (grabbed, image) = self.cam.read()
            if not grabbed:
                break
            #image = frame.array
            height, width,channel = image.shape
            size=90
            frame_crop = image[0:height,width/2-size:width/2+size]
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
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
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

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray
            #cv2.imshow('lk_track', vis)

            deg=np.zeros((0,0))
            now_x=np.zeros((0,0))
            for tracks in range (0,(len(self.tracks)-1)):
                test=np.squeeze(np.asarray(self.tracks[tracks:tracks+1]))   #make one of the track point as an array, and reduce its dimension
                if (test[-1:0]-test[0:0]) == 0 or len(test)<=5:             #check to prevent error,check not to use new track point
                    continue
                now_x=np.append(now_x,test[-1,0])#track newest point
                deg=np.append(deg,math.degrees(
                    math.atan((test[-1,1]-test[0,1])/(test[-1,0]-test[0,0]))))#find the gradient of the points, then convert to degree and append
            #print now_x
            #print deg           #now this variable contains array of degrees correspond to each track point.
            print np.std(deg)   #if all track point move approximately the same gradient, the camera is perpendicular to surface.
            if len(now_x) != len(self.previous_x):
                self.previous_x=now_x
            else:
                diff = self.previous_x-now_x
                #print self.absolute_x
            cv2.imshow('lk_track', vis)
            #print radian
            #print testc[:,:,1]

            '''if len(test) is 0:
                cumm=0
            else:
                (point,cumm,axis)=test.shape
                print test[:,-1,:]
            if cumm>2:
                diff = test[-1,-2,:]-test[-1,-1,:]
                self.absolute += diff
                #print diff,self.absolute
                #plt.plot(absolute[0],absolute[1])
                #plt.show()
            # clear the stream in preparation for the next frame
            #self.rawCapture.truncate(0)'''

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27 or ch ==ord("q"):
                break

def main():
    #import sys
    #try: video_src = sys.argv[1]
    #except: video_src = 0

    #print __doc__
    #App().distance()
    App().run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
