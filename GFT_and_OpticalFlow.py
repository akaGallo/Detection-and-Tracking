import cv2
import numpy as np
from Utility import drawKeypoints

MAX_FRAMES = 2000
TRACKED_POINTS = 5 # Minimum number of remaining tracked points along frames

def GoodFeatToTrack_OptFlow_algorithm(video):
    # Open first frame
    ret, frame = video.read()
    previousFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create parameters for goodFeaturesToTrack algorithm
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Create parameters for calcOpticalFlowPyrLK algorithm
    lk_params = dict( winSize  = (30, 30),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    color = np.random.randint(0,255,(100,3))

    # Detect keypoints with goodFeaturesToTrack algorithm
    corners = cv2.goodFeaturesToTrack(previousFrame, mask = None, **feature_params)
    mask = np.zeros_like(previousFrame)
 
    for t in range(MAX_FRAMES):
        # Open next frame
        ret, frame = video.read()

        if not ret:
            break
        
        actualFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
        # Track corners in the actual frame
        nextPoints, status, error = cv2.calcOpticalFlowPyrLK(previousFrame, actualFrame, corners, None, **lk_params)

        if(sum(status) < TRACKED_POINTS):
            # Detect and track new keypoints
            corners = cv2.goodFeaturesToTrack(actualFrame, mask = None, **feature_params)
            nextPoints, status, error = cv2.calcOpticalFlowPyrLK(previousFrame, actualFrame, corners, None, **lk_params)
            mask = np.zeros_like(previousFrame)
            
        # Draw tracking line in the actual frame
        newPoints = nextPoints[status == 1]
        oldPoints = corners[status == 1]
        trackedFrame = drawKeypoints(newPoints, oldPoints, actualFrame, mask, color)

        # Show the output for detection + tracking algorithm
        cv2.imshow("Good Feature To Track + Optical Flow", trackedFrame)
   
        # Update the previous frame and previous points
        previousFrame = actualFrame.copy()
        corners = newPoints.reshape(-1, 1, 2)

        # Exit if 'q' pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the video and destroy all
    video.release()
    cv2.destroyAllWindows()