import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)


folderPath = "FingerImages"
myList = os.listdir(folderPath)
myList.sort()
print(myList)
pTime = 0
overlayList = []
for imagePath in myList:
    image = cv2.imread(f'{folderPath}/{imagePath}')
    overlayList.append(image)  # import all the images into the program
#print(overlayList)
detector = htm.handDetector(detectionConfidence=0.75, trackConfidence=0.75)
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findhands(img)
    lmList = detector.findPosition(img, draw=False)
    #print(lmList)
    if len(lmList) != 0:
        fingers = []
        #Thumb closed or open
        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:  #If its open
            fingers.append(1)
        else:
            fingers.append(0)
        #Other 4 fingers closed or open
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #Count number of fingers open
        totalFingers = fingers.count(1)
        #Change image accordingly
        h, w, c = overlayList[0].shape
        img[0:200, 0:200] = overlayList[totalFingers-1]

        cv2.rectangle(img, (20, 225), (170,425), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (40,375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255,0,0), 25)
    # FPS Calculation
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.imshow("Image", img)
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    cv2.waitKey(1)
