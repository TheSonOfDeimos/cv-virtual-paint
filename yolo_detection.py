import numpy as np
import cv2
from imutils.video import FPS
import keyboard


def get_class_with_max_confidence(outputs, conf_threshold):
    d = {}
    for i in outputs:
        part = i[:, 5:]  # get confidences of classes
        index = np.unravel_index(np.argmax(part), part.shape)  # index of max confidence in output layer
        d[index] = part[index]

    if max(d.values()) < conf_threshold:  # max confidence is less than threshold
        return None
    else:
        return int(max(d, key=d.get)[1])  # class ID of max confidence


def make_prediction(net, layer_names, image, conf_threshold):
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)
    return get_class_with_max_confidence(outputs, conf_threshold)


def yolo_detection(gesture_lock):
    labels = open('model/_darknet.labels').read().strip().split('\n')
    cfg, weights = 'model/custom-yolov4-detector.cfg', 'model/yolov4-hand-gesture.weights'
    net = cv2.dnn.readNetFromDarknet(cfgFile=cfg, darknetModel=weights)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(0)
    address = "http://192.168.1.193:8080/video"
    cap.open(address)
    fps = FPS().start()

    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print('Video file finished.')
            break

        class_id = make_prediction(net, layer_names, image, conf_threshold=0.9)
        if class_id is not None:  # keep previous state, if None
            gesture_lock.set_gesture(labels[class_id])

        fps.update()
        if keyboard.is_pressed('esc'):  # can't use cv's waitKey, cuz no window
            break

    fps.stop()
    print("Mean fps for detection:", round(fps.fps(), 2))
    cap.release()
