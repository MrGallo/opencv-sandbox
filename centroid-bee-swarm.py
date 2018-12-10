import cv2
from time import perf_counter as pc

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -7)

# MOG - has some noise
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

# GMG - removes noise, slower.
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg_gmg = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
    start = pc()
    _, frame = cap.read()

    # fg_mask_mog = fgbg.apply(frame)  # 8 ms

    fg_mask_gmg = fgbg_gmg.apply(frame)  # 35 ms
    fg_mask_gmg = cv2.morphologyEx(fg_mask_gmg, cv2.MORPH_OPEN, kernel)

    # convert the grayscale image to binary image
    _, thresh = cv2.threshold(fg_mask_gmg,127,255,0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # calculate moments for each contour
       M = cv2.moments(c)

       # calculate x,y coordinate of center
       cX = int(M["m10"] / M["m00"])
       cY = int(M["m01"] / M["m00"])
       cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
       cv2.putText(frame, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


    cv2.imshow("frame", frame)
    # cv2.imshow("fg_mask_mog", fg_mask_mog)
    # cv2.imshow("fg_mask_gmg", fg_mask_gmg)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    length = int((pc() - start) * 1000)  # sec to millis
    print(f"{length} ms")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
