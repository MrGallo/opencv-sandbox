import cv2
import numpy as np


img = cv2.imread('original.JPG')
resized = cv2.resize(img, (600, 400))

# convert to gray
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# threshold the gray
ret,thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

# apply closing to fill in holes
kernel = np.ones((5, 5), dtype=np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# find contours
_, contours, _ = cv2.findContours(closing, 1, 2)

# for every contour, draw a bounding box
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(resized, (x, y), (x+w, y+h),(0, 255, 0), 2)

while True:
    cv2.imshow("closing", closing)
    cv2.imshow("resized", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cv2.destroyAllWindows()