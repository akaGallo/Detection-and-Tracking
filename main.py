import cv2
import sys

from ORB_and_OpticalFlow import ORB_OptFlow_algorithm
from SIFT_and_OpticalFlow import SIFT_OptFlow_algorithm
from GFT_and_OpticalFlow import GoodFeatToTrack_OptFlow_algorithm

if __name__ == '__main__':
    for i in range(3):
        # Read video
        video = cv2.VideoCapture("../SecondAssignment/Material/Contesto_industriale1.mp4")

        # Exit if video not opened
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        # Check first frame
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()
        
        if i == 0:
            GoodFeatToTrack_OptFlow_algorithm(video)
        elif i == 1:
            SIFT_OptFlow_algorithm(video)
        else:
            ORB_OptFlow_algorithm(video)
