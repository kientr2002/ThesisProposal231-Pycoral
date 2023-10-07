import re
import os
import cv2
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import pycoral as pcr

from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common
from pycoral.adapters import classify
from imutils.video import FPS


# the TFLite converted to be used with edgetpu
modelPath = 'model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = 'labels.txt'

#some value to insert the Text to CV2

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (0, 0, 0)  # Text color (Yellow)
font_thickness = 1

# Create a folder "captured_imaged"
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)

#The conditional of detection objects

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
    fps = FPS().start()
    frame_times = []
    CPU_Usages = []
    pcr.adapters.classify
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Flip image so it matches the training input
        frame = cv2.flip(frame, 1)

        # Classify and display image
        results = classifyImage(interpreter, frame)
        print(f'Label: {labels[results[0].id]}, Score: {results[0].score}')
        text = f'Label: {labels[results[0].id]}, Score: {results[0].score:.2f}'

        fps.update()
        fps.stop()
        frame_times.append(round(fps.fps(),2))
        fps_on_frame = f'FPS: {round(fps.fps(),2)}'
        cpu_usages_on_frame = f'CPU Usages: {psutil.cpu_percent()} %'
        CPU_Usages.append(psutil.cpu_percent())
        cv2.putText(frame, text, (10, 50), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, fps_on_frame + ' ' + cpu_usages_on_frame, (10, 70), font, font_scale, font_color, font_thickness)
        cv2.imshow('Camera AI using coral', frame)
        # CPU Usage
        print(f'CPU Usages: {psutil.cpu_percent()} %')
        print(f'FPS: {fps.fps():.2f}')
        if results[0].score > 0.8:  # Kiểm tra xem có đối tượng được phát hiện hay không
            # Tạo tên tệp tin duy nhất dựa trên thời gian hiện tại
            timestamp = int(time.time())
            image_filename = os.path.join(output_dir, f'captured_{timestamp}.jpg')

            # Lưu ảnh từ frame
            cv2.imwrite(image_filename, frame)
            print(f'Đã chụp ảnh và lưu tại: {image_filename}')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        key = cv2.waitKey(1) & 0xFF
        # Press "Q" Button to shut down the Program
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    plt.plot(frame_times)
    plt.xlabel('Khung hình')
    plt.ylabel('FPS')
    plt.title('Biểu đồ FPS')
    plt.show()
if __name__ == '__main__':
    main()
