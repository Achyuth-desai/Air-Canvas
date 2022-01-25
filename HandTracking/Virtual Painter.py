import cv2
import numpy as np
import mediapipe as mp
import time
import os
import HandTrackingModule as htm

#############################
brushThickness = 5
eraserThickness = 50
#############################
detector = htm.handDetector(detectionConfidence=0.85)
folderPath = "Header"
myList = os.listdir(folderPath)
myList.sort()  # File Names
overlayList = []  # Images that are read by the program in RGB form
# Import all the images into the program
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
xp, yp = 0, 0
# Create a image canvas to draw
imageCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # 1. Import images
    success, img = cap.read()
    img = cv2.flip(img, 1)
    # 2. Find Hand Landmarks
    img = detector.findhands(img)
    landmarkList = detector.findPosition(img, draw=False)
    if len(landmarkList) != 0:
        x1, y1 = landmarkList[8][1:]  # X and Y co-ordinates of tip of index finger
        x2, y2 = landmarkList[12][1:]  # X and Y co-ordinates of tip of middle finger
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        # 4. If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print("Selection mode")
            # Checking for the click
            if y1 < 104:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (0, 0, 255)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing mode - One finger is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing mode")
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
            cv2.line(imageCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imageCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv) #Erasing part
    img = cv2.bitwise_or(img, imageCanvas)

    # Setting Header Image with size
    img[0:104, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imageCanvas)
    cv2.waitKey(1)
