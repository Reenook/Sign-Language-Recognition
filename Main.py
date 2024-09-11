import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300

labels = ["I Love you", "Call me", "Peace"]

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Ensure cropping area is within image boundaries
        height, width, _ = img.shape
        x_start = max(x - offset, 0)
        y_start = max(y - offset, 0)
        x_end = min(x + w + offset, width)
        y_end = min(y + h + offset, height)

        if x_start >= x_end or y_start >= y_end:
            print("Invalid crop region!")
            continue

        imgCrop = img[y_start:y_end, x_start:x_end]

        if imgCrop.size == 0:
            print("Empty crop area!")
            continue

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            if wCal <= 0:
                print("Invalid calculated width!")
                continue
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            if hCal <= 0:
                print("Invalid calculated height!")
                continue
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Determine text size and adjust padding
        if index is not None:
            text = labels[index]
            font_scale = 1.7
            font_thickness = 2
            font = cv2.FONT_HERSHEY_COMPLEX
            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            text_width, text_height = text_size

            # Padding based on text size
            padding = 10
            text_area_height = text_height + padding * 2
            text_area_width = text_width + padding * 2

            # Adjust rectangle size based on text area
            rect_x1 = x - offset
            rect_y1 = y - offset - text_area_height
            rect_x2 = x - offset + text_area_width
            rect_y2 = y - offset

            cv2.rectangle(imgOutput, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, text, (x - offset + padding, y - offset - padding), font, font_scale,
                        (255, 255, 255), font_thickness)

        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
