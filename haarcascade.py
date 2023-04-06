import cv2
import numpy as np

cap = cv2.VideoCapture('cctv_video.mp4')

roi = [(50, 50), (450, 350)]

fgbg = cv2.createBackgroundSubtractorMOG2()

heatmap = np.zeros((400, 500), dtype=np.float32)

color_map = cv2.COLORMAP_HOT

cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

cv2.namedWindow('Video Frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Video Frame', 1200, 900)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    frame_roi = frame[y1:y2, x1:x2]

    fgmask = fgbg.apply(frame_roi)

    thresh = cv2.threshold(fgmask, 127, 255, cv2.THRESH_BINARY)[1]

    gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=3)

    for (x, y, w, h) in detections:
        cv2.rectangle(frame_roi, (x, y), (x+w, y+h), (0, 255, 0), 2)

    heatmap[y1:y2, x1:x2] += thresh.astype(np.float32)

    heatmap_colored = cv2.applyColorMap((heatmap / heatmap.max() * 255).astype(np.uint8), color_map)

    heatmap_colored[..., 0] = 0
    heatmap_colored[..., 1] = 0

    alpha = 0.5
    frame_merged = cv2.addWeighted(frame_roi, alpha, heatmap_colored[y1:y2, x1:x2], 1-alpha, 0)

    cv2.imshow('Video Frame', frame_merged)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()