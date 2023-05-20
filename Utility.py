import cv2

def keypoint_into_coordinates(corners, keypoint):
    for kp in keypoint:
        x,y = kp.pt
        point = (x, y)
        corners.append(point)

def drawKeypoints(newPoints, oldPoints, actualFrame, mask, color):
    for i, (new, old) in enumerate(zip(newPoints, oldPoints)):
        a, b = new.ravel()
        a, b = int(a), int(b)
        c, d = old.ravel()
        c, d = int(c), int(d)
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(), 6)
        frame = cv2.circle(actualFrame, (a,b), 8, color[i].tolist(), -1)
    trackedFrame = cv2.add(frame, mask)
    return trackedFrame