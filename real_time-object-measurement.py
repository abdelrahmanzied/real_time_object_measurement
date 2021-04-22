#Import libraries
import cv2
import numpy as np


# Length Function
# Using euclides equation to find the real length
def findLength(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5


# Reorder Function
#Order points of contour  (top-left, top-right, down-left, down-left)
def reorderPoints(box_points):
    ordered_points = np.zeros_like(box_points)
    box_points = box_points.reshape((corners, 2))
    add = box_points.sum(1)
    ordered_points[0] = box_points[np.argmin(add)]
    ordered_points[3] = box_points[np.argmax(add)]
    diff = np.diff(box_points, axis=1)
    ordered_points[1] = box_points[np.argmin(diff)]
    ordered_points[2] = box_points[np.argmax(diff)]
    return ordered_points


# GetContours Function
# Detect contours, draw them, and calculate the real width, height and area.
def getContours(img,imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt) # the area of contour in px.
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # values from trackbars.
        areaMin = cv2.getTrackbarPos("Area", "Parameters") * 1000  # area from trackbar in m² * 30000 as a scaler for area.
        distance = cv2.getTrackbarPos("Distance", "Parameters")
        scale = cv2.getTrackbarPos("Scale", "Parameters")

        # draw and measure the contour that depends on the area from user and 4 corners of the shape
        if area>areaMin and len(approx)==corners:
            x, y, w, h = cv2.boundingRect(approx)
            app = np.array(approx)
            ordered_points = reorderPoints(app) # order points

            # alpha is the scaler of length in pixels depends on the distance of the camera
            # width in px = 30, distance = 1m, scale = 6 --> alpha = 6/1 = 6 --> width in cm = 30 / 6 = 5cm
            # if the distance increases the pixels decreases in the frame so to keep the same ratio aplha should be decreased
            # scale is fixed depends on the frame size
            # width in px = 15, distance = 2m, scale = 6 --> alpha = 6/2 = 3 --> width in cm = 15 / 3 = 5cm
            alpha = scale/distance


            real_w = round((findLength(ordered_points[0][0] / alpha , ordered_points[1][0] / alpha )), 2)
            real_h = round((findLength(ordered_points[0][0] / alpha, ordered_points[2][0] / alpha)), 2)
            area_m2 = round((real_w/100) * (real_h/100), 2) # divide width and height to get area in squared meter.
            print(f"Width: {real_w}cm, Height: {real_h}cm, Area: {area_m2}m²")


            # draw width and height of the contour
            cv2.arrowedLine(imgContour, (ordered_points[0][0][0], ordered_points[0][0][1]), (ordered_points[1][0][0], ordered_points[1][0][1]), color, 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContour, (ordered_points[0][0][0], ordered_points[0][0][1]), (ordered_points[2][0][0], ordered_points[2][0][1]), color, 3, 8, 0, 0.05)


            # print the width, height, and area of the contour
            cv2.putText(imgContour, '{}cm'.format(real_w), (x + 20, y - 10), font, 1, color, 2)
            cv2.putText(imgContour, '{}cm'.format(real_h), (x - 100, y + h // 2), font, 1, color, 2)
            cv2.putText(imgContour, 'Area: {}m2'.format(area_m2), (x + w//10, y + h//2), font, 1, (0,0,255), 2)


# Stack images function
# group the original image, canny image, eroded image and the result image.
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# Trackbars
def empty(x):
    pass
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",64,255,empty)
cv2.createTrackbar("Threshold2","Parameters",19,255,empty)
cv2.createTrackbar("Area","Parameters",1,100,empty)  # area = the area of the object to filter objects
cv2.createTrackbar("Distance","Parameters",1, 5,empty)      # distance = the distance in meter between the object and the camera
cv2.createTrackbar("Scale","Parameters",6, 25,empty)        # scale = the number of pixels in the 1cm --> 1cm = 6px


# Video Properties
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


# Initial values
corners = 4     # the corners of the contour
color = (0, 255, 0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


# Main App
while True:
    success, img = cap.read()
    imgContour = img.copy()
    imgBlur = cv2.GaussianBlur(img, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThre = cv2.erode(imgDil, kernel, iterations=2)

    getContours(imgThre, imgContour)

    imgStack = stackImages(0.8,([img,imgCanny], [imgThre,imgContour]))
    cv2.imshow("Result", imgStack)
    cv2.imshow("Measurement", imgContour)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break