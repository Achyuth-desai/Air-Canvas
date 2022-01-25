import cv2
import mediapipe as mp
import time

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mphands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils
        self.hands = self.mphands.Hands(self.mode, self.maxHands,
                                        self.detectionConfidence, self.trackConfidence)
        self.tipIds = [4, 8, 12, 16, 20]


    def findhands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            # In case of multiple hands, we have to draw landmarks for each hand. So we use for loop:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mphands.HAND_CONNECTIONS)  # Red dots on hands
        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # X and Y coordinate of each landmark on hand in px
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255,0,255), cv2.FILLED)
        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    while True:
        successs, img = cap.read()
        img = detector.findhands(img)
        lmList  = detector.findPosition(img);
        if len(lmList) !=0:
            print(lmList[0])

        # FPS Calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.namedWindow('Image', flags=cv2.WINDOW_GUI_NORMAL)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if  __name__ == "__main__":
    main()