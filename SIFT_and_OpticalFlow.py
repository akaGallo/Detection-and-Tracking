import cv2
import numpy as np
from Utility import keypoint_into_coordinates, drawKeypoints

NUMBER_FEATURES = 50
MAX_FRAMES = 2000
TRACKED_POINTS = 20 # Minimum number of remaining tracked points along frames

def SIFT_OptFlow_algorithm(video):
    sift = cv2.SIFT_create(NUMBER_FEATURES)

    # Open first frame
    ret, frame = video.read()
    previousFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create parameters for calcOpticalFlowPyrLK algorithm
    lk_params = dict( winSize = (30, 30),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0, 255, (100, 3))

    # Find keypoints of first frame and obtain their coordinates
    keypoint, descriptor = sift.detectAndCompute(previousFrame, None)
    corners = []
    keypoint_into_coordinates(corners, keypoint)
    corners = np.float32(corners)
    mask = np.zeros_like(previousFrame)

    for t in range(MAX_FRAMES):
        # Open next frame
        ret, frame = video.read()
        
        if not ret:
            break

        actualFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        # Detect and track keypoints in the actual frame
        keypoint, descriptor = sift.detectAndCompute(actualFrame, None)
        nextPoints, status, error = cv2.calcOpticalFlowPyrLK(previousFrame, actualFrame, corners, None, **lk_params)

        if(sum(status) < TRACKED_POINTS):
            corners = []
            keypoint_into_coordinates(corners, keypoint)
            corners = np.float32(corners)
            
            nextPoints, status, error = cv2.calcOpticalFlowPyrLK(previousFrame, actualFrame, corners, None, **lk_params)
            mask = np.zeros_like(previousFrame)

        # Draw tracking line in the actual frame
        newPoints = nextPoints
        oldPoints = corners
        trackedFrame = drawKeypoints(newPoints, oldPoints, actualFrame, mask, color)

        # Match the best (according to the distance) detected features along two consecutive frames
        if(t > 0):
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = True)
            matches = bf.match(previous_descriptor, descriptor)
            matches = sorted(matches, key = lambda x:x.distance)

            finalFrame = cv2.drawMatches(previousFrame, previous_keypoint, trackedFrame, keypoint, 
                                            matches[:20], trackedFrame, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Show the output for detection + tracking algorithm
            cv2.imshow('SIFT + Optical Flow', finalFrame)

        # Update the previous frame and previous points/keypoints
        previous_keypoint, previous_descriptor = keypoint, descriptor
        previousFrame = actualFrame.copy()
        corners = newPoints

        # Exit if 'q' pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video and destroy all
    video.release()
    cv2.destroyAllWindows()