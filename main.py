import re
import os
import cv2
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import pycoral as pcr
import threading

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
font_color = (55, 85, 255)  # Text color (Yellow)
font_thickness = 1

# Create a folder "captured_imaged"
output_dir = 'captured_images'
os.makedirs(output_dir, exist_ok=True)

# Identification
counter = 0  # Count Images with the detection over 0.8
now_Object = ""
previous_Object = ""
label_now = ""
label_previous = ""
time_detection_flag = 0
# Biến đếm giây
seconds = 0

#The conditional of detection objects

# This function takes in a TFLite Interptere and Image, and returns classifications
def classifyImage(interpreter, image):
    size = common.input_size(interpreter)
    common.set_input(interpreter, cv2.resize(image, size, fx=0, fy=0,
                                             interpolation=cv2.INTER_CUBIC))
    interpreter.invoke()
    return classify.get_classes(interpreter)
def count_seconds():
    global seconds
    while True:
        time.sleep(1)
        seconds += 1

def Counter(input):
    global counter,now_Object,previous_Object, label_now, label_previous, time_detection_flag
    if counter == 0:
        time_detection_flag = 0
        now_Object = input
        counter = counter + 1
    elif counter > 0 and counter < 5:
        previous_Object = now_Object
        now_Object = input
        if now_Object == previous_Object:
            counter = counter + 1
        else:
            previous_Object = ""
            now_Object = ""
            counter = 0
    else:
        previous_Object = now_Object
        now_Object = input
        if now_Object == previous_Object:
            label_previous = label_now
            label_now = now_Object
            time_detection_flag = 1
        now_Object = ""
        previous_Object = ""
        counter = 0


count_thread = threading.Thread(target=count_seconds)
count_thread.daemon = True
count_thread.start()

def main():
    global counter, now_Object, previous_Object, label_now
    # Load your model onto the TF Lite Interpreter
    interpreter = make_interpreter(modelPath)
    interpreter.allocate_tensors()
    labels = read_label_file(labelPath)

    cap = cv2.VideoCapture(0)
    fps = FPS().start()
    frame_times = []
    CPU_Usages = []
    Scores = []
    Times = []
    Label_of_Times = []
    count_Times_data = []
    count_Time_data = 0
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
        print(f'Label detected: {label_now} and counter: {counter}')
        text = f'Label: {labels[results[0].id]}, Score: {results[0].score:.2f}'

        fps.update()
        fps.stop()
        frame_times.append(round(fps.fps(),2))
        Scores.append(round(results[0].score, 4))
        fps_on_frame = f'FPS: {round(fps.fps(),2)}'

        cpu_usages_on_frame = f'CPU Usages: {psutil.cpu_percent()} %'
        CPU_Usages.append(psutil.cpu_percent())
        cv2.putText(frame, label_now, (10, 50), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, text, (10, 70), font, font_scale, font_color, font_thickness)
        cv2.putText(frame, fps_on_frame + ' ' + cpu_usages_on_frame, (10, 90), font, font_scale, font_color, font_thickness)
        cv2.imshow('Camera AI using coral', frame)
        # CPU Usage
        print(f'CPU Usages: {psutil.cpu_percent()} %')
        print(f'FPS: {fps.fps():.2f}')
        if results[0].score > 0.8:  # Kiểm tra xem có đối tượng được phát hiện hay không
            # Tạo tên tệp tin duy nhất dựa trên thời gian hiện tại
            #Counter Detection
            Counter(labels[results[0].id])
        else:
            counter = 0
            now_Object = ""
            previous_Object = ""
            label_now = ""
        if time_detection_flag == 1 and label_previous != label_now:
            Label_of_Times.append(label_now)
            Times.append(seconds)
            count_Time_data = count_Time_data + 1
            count_Times_data.append(count_Time_data)
        timestamp = int(time.time())
        image_filename = os.path.join(output_dir, f'captured_{timestamp}.jpg')
        # Lưu ảnh từ frame
        cv2.imwrite(image_filename, frame)
        print(f'Đã chụp ảnh và lưu tại: {image_filename}')
        print('-------------------------------------------------------------')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        key = cv2.waitKey(1) & 0xFF
        # Press "Q" Button to shut down the Program
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    #FPS
    plt.plot(frame_times)
    plt.xlabel('Khung hình')
    plt.ylabel('FPS')
    plt.title('Frame per Second')
    plt.show()
    #CPU Usages
    plt.plot(CPU_Usages)
    plt.xlabel('Khung hình')
    plt.ylabel('%')
    plt.title('CPU Usages')
    plt.show()
    #Scores
    plt.plot(Scores)
    plt.xlabel('Khung hình')
    plt.ylabel('Score')
    plt.title('Accuracy')
    plt.axhline(y=0.8, color='red', linestyle='--', label='y = 0.8')
    plt.show()
    #Time to detection
    plt.scatter(count_Times_data, Times)
    plt.xlabel('Khung hình')
    plt.ylabel('second')
    plt.title('Time Detection')
    # Vẽ chiếu dọc từ các điểm đến trục x
    for i in range(len(count_Times_data)):
        plt.vlines(count_Times_data[i], ymin=0, ymax=Times[i], colors='blue', linestyle='--')

    # Vẽ chiếu ngang từ các điểm đến trục y
    for i in range(len(Times)):
        color = "black"
        if Label_of_Times[i] == "Nothing":
            color = 'black'
        elif Label_of_Times[i] == "Red":
            color = 'red'
        elif Label_of_Times[i] == "Yellow":
            color = 'yellow'
        elif Label_of_Times[i] == "Green":
            color = 'blue'
        plt.hlines(Times[i], xmin=0, xmax=count_Times_data[i], colors=color, linestyle='--')
    # Hiển thị giá trị cụ thể trên từng điểm
    for i in range(len(count_Times_data)):
        if Times[i] != Times[i - 1]:
            plt.annotate(f'({count_Times_data[i]}, {Times[i]})', (count_Times_data[i], Times[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

if __name__ == '__main__':
    main()