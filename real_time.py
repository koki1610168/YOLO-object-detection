import imutils
import cv2
import numpy as np
import argparse
import time
import os

#Argument Setting - Passed in the terminal
ap = argparse.ArgumentParser()
ap.add_argument('-y', '--yolo', required=True, help='base path to YOLO directory')
ap.add_argument('-c', '--confidence', type=float, default=0.5,
        help='minimum probability to filter weak detections')
ap.add_argument('-t', '--threshold', type=float, default=0.3, 
        help='threshold when applying non-maxima suppression')

args = vars(ap.parse_args())

#Load COCO labels
labelsPath = os.path.sep.join([args['yolo'], 'coco.names'])
LABELS = open(labelsPath).read().strip().split('\n')

#Random Color for labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

weightsPath = os.path.sep.join([args['yolo'], 'yolov3.weights'])
configPath = os.path.sep.join([args['yolo'], 'yolov3.cfg'])

print("[Processing] loading YOLO from disk ... ")

#Load pre-trained model from YOLO
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#get layer names. ex) conv_0, relu_0
ln = net.getLayerNames()
#extract unconnected layers: yolo82, 94, 106
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


vs = cv2.VideoCapture(1)
(W, H) = (None, None)

if not vs.isOpened():
    print("Could not open video device")

while True:
    ret, frame = vs.read()
    #cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    #build a blob image - performing mean subtraction, scaling, etc.
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes = []
    confidences = []
    classIDs = []

    #loop over each layer outputs
    for output in layerOutputs:
        for detection in output:
            #extract the highest score from each detection and throw it to confidence
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            #rule out weak confidence in order to get more accurate data
            if confidence > args['confidence']:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args['confidence'], args['threshold'])

    #DRAW BOXES AND TEXT!!!
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('frame', frame)

vs.release()
cv2.destroyAllWindows()
#Capture video from the camera
