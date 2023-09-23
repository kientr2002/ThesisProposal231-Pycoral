import re
import os
import cv2
import numpy as np
import time
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify


# the TFLite converted to be used with edgetpu
modelPath = 'model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = 'labels.txt'

#some value to insert the Text to CV2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 255, 255)  # Text color (Yellow)
font_thickness = 1


# This function takes in a TFLite Interptere and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return classify.get_classes(interpreter)

def main():
    # Load your model onto the TF Lite Interpreter
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not ret:
            break

        # Flip image so it matches the training input
        frame = cv2.flip(frame, 1)

        # Classify and display image
        results = classifyImage(interpreter, frame)
        print(f'Label: {labels[results[0].id]}, Score: {results[0].score}')
        text = f'Label: {labels[results[0].id]}, Score: {results[0].score:.2f}'
        cv2.putText(frame, text, (10, 50), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, f'FPS: {fps}', (10, 30), font, font_scale, font_color, font_thickness)
        cv2.imshow('Camera AI using coral', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        key = cv2.waitKey(1) & 0xFF
        # Press "Q" Button to shutdown the Program
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
