import os
import sys
import cv2 as cv
import copy
import csv
import itertools
import numpy as np
import time

# GUI
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import uic

# TV control
from pywebostv.connection import *
from pywebostv.controls import *

import mediapipe as mp
import tensorflow as tf

# from model import HandsignClassifier

# from LG_REMOTE.oop_remote import LG_Remote

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path,relative_path)

class HandsignClassifier():
    def __init__(self,model_path=resource_path("handsign_classifierv4.tflite"),num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()


    def __call__(self,landmark_list):
        input_details_tensor_index = self.input_details[0]['index']

        self.interpreter.set_tensor(input_details_tensor_index,
                                    np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        if np.squeeze(result)[result_index] > 0.85:
            return result_index


class LG_Remote:
    def select_mode(self, key, mode):
        number = -1
        if 48 <= key <= 57:
            number = key - 48
        if key == 110:  # n - for normal screen
            mode = 0
        if key == 104:  # h - for saving hand sign
            mode = 1

        return number, mode

    def calc_landmark_list(self, img, landmrks):
        img_width, img_height = img.shape[1], img.shape[0]

        landmark_list = []

        for id, landmark in enumerate(landmrks.landmark):
            landmark_x = min(int(landmark.x * img_width), img_width - 1)
            landmark_y = min(int(landmark.y * img_height), img_height - 1)

            landmark_list.append([landmark_x, landmark_y])

        return landmark_list

    def calc_bounding_rect(self, img, landmrks):
        landmark_list = self.calc_landmark_list(img, landmrks)
        landmrk_array = np.array(landmark_list).reshape(21, 2)

        x, y, w, h = cv.boundingRect(landmrk_array)  # boundingRect require numpy array

        return [x, y, x + w, y + h]

    def pre_process_landmarks(self, landmrk_list):
        temp_landmrk_list = copy.deepcopy(landmrk_list)
        base_x, base_y = 0, 0
        for id, landmrk in enumerate(temp_landmrk_list):
            if id == 0:
                base_x, base_y = landmrk[0], landmrk[1]

            temp_landmrk_list[id][0] = temp_landmrk_list[id][0] - base_x
            temp_landmrk_list[id][1] = temp_landmrk_list[id][1] - base_y

        # convert to a one dimensional list
        temp_landmrk_list = list(itertools.chain.from_iterable(temp_landmrk_list))

        # finding maximum value
        max_value = max(list(map(abs, temp_landmrk_list)))

        # normalization
        def normalize(n):
            return n / max_value

        temp_landmrk_list = list(map(normalize, temp_landmrk_list))

        return temp_landmrk_list

    def logging_csv(self, number, mode, landmark_lst):
        if mode == 0:
            pass
        if mode == 1 and (0 <= number <= 9):
            with open("../model/Handsign_classifier/handsign_classifier_datav4.csv", 'a', newline='') as f:
                write = csv.writer(f)
                write.writerow([number, *landmark_lst])

        return

    def draw_landmarks(self, image, landmark_point):
        if len(landmark_point) > 0:
            # Thumb
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                    (255, 255, 255), 2)

            # Index finger
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                    (255, 255, 255), 2)

            # Middle finger
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                    (255, 255, 255), 2)

            # Ring finger
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                    (255, 255, 255), 2)

            # Little finger
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                    (255, 255, 255), 2)

            # Palm
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                    (255, 255, 255), 2)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (0, 0, 0), 6)
            cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                    (255, 255, 255), 2)

        # Key Points
        for index, landmark in enumerate(landmark_point):
            if index == 0:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 1:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 2:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 3:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 4:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 5:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 6:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 7:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 8:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 9:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 10:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 11:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 12:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 13:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 14:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 15:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 16:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
            if index == 17:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 18:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 19:
                cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
            if index == 20:
                cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                          -1)
                cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

        return image

    def draw_bounding_rect(self, use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image

    def draw_info(self, image, fps, mode, number):
        cv.putText(image, str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                   1.0, (0, 255, 0), 2, cv.LINE_AA)

        if mode == 1:
            cv.putText(image, "MODE : Logging Hand Sign", (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
            if 0 <= number <= 9:
                cv.putText(image, "NUM:" + str(number), (10, 110),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                           cv.LINE_AA)
        return image

    def draw_info_text(self, image, brect, handedness, hand_sign_text):

        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                     (0, 0, 0), -1)

        info_text = handedness.classification[0].label[0:]
        if hand_sign_text != "":
            info_text = info_text + ':' + hand_sign_text
        cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        return image

# class FeatureWindow(QDialog):
#     def __init__(self):
#         super(FeatureWindow,self).__init__()
#         uic.loadUi(resource_path("features.ui"), self)
#         self.pushButton_2.clicked.connect(self.gotoMainScreen)
#         self.show()
#         self.feature_pic = os.listdir("fet_pics") # 208x167
#
#         self.feature_pic = sorted(self.feature_pic,key=lambda x : int(x[:x.find(")")]),reverse=False)
#         # print(self.feature_pic)
#
#         labels = [self.label,self.label_2,self.label_3,self.label_4,self.label_5,self.label_6,self.label_7,self.label_8,self.label_9,self.label_10]
#
#         for ind,_ in enumerate(labels):
#             # if ind < 6:
#             pic_path = os.path.join("fet_pics",self.feature_pic[ind])
#             self.fetqpixmap = QPixmap(resource_path(pic_path))
#             labels[ind].setPixmap(self.fetqpixmap)
#
#         # self.label.resize(232, 160)
#         # self.label.setPixmap(self.fetqpixmap)
#         # print(self.feature_pic)
#
#
#     def gotoMainScreen(self):
#         mainwindow = MainWindow()
#         widget.addWidget(mainwindow)
#         widget.setCurrentIndex(widget.currentIndex()+1)


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        uic.loadUi(resource_path("remote2.ui"),self)

        self.setWindowTitle("LG Ai Remote")

        self.setWindowIcon(QIcon(resource_path("remote_logo.jpg")))

        self.show()

        self.ip = None

        self.qpixmap = QPixmap(resource_path("re_plch.jpg"))
        self.store = {}

        width = 755
        height = 810
        # setting the maximum size
        self.setMaximumSize(width, height)


        # self.label.setStyleSheet("border :2px solid black;")

        self.label.setPixmap(self.qpixmap)

        self.pushButton.setCheckable(True)

        self.pushButton.clicked.connect(self.on_off_toggle)

        # self.actiontour_2.triggered.connect(self.gotoFeatureScreen)
        # self.pushButton_2.clicked.connect(self.Connect2TV)

    # def Connect2Tv(self):
    #     print(self.lineEdit.text())

    # def Update_ip(self,ip):

    # def gotoFeatureScreen(self):
    #     fet_screen = FeatureWindow()
    #     widget.addWidget(fet_screen)
    #     widget.setCurrentIndex(widget.currentIndex()+1)

    def ImageUpdateSlot(self,Img):
        self.label.setPixmap(QPixmap.fromImage(Img))

    def HandSignDetection(self,txt):
        self.textBrowser.setText(txt)

    def ResponseScreen(self,txt):
        self.textBrowser_2.setText(txt)


    def notify(self,client):
        Sys = SystemControl(client)
        Sys.notify("LG AI remote is connected.")

    def update_store(self,st_dict):
        self.store = st_dict

    def add_default_img(self):
        # if self.pushButton.isChecked():
        #     print(self.pushButton.isChecked())
        self.label.setPixmap(self.qpixmap)


    def on_off_toggle(self):
        if self.pushButton.isChecked():
            self.ip = self.lineEdit.text()

            if len(self.ip) != 0:
                self.worker1 = Worker1(store = self.store,ip=self.ip)
            else:
                self.worker1 = Worker1(store=self.store)

            self.worker1.start()
            self.worker1.store_dict.connect(self.update_store)
            self.worker1.connection_response_txt.connect(self.ResponseScreen)
            self.worker1.ImageUpdate.connect(self.ImageUpdateSlot)
            self.worker1.SignTxtUpdate.connect(self.HandSignDetection)
            self.pushButton.setStyleSheet("background-color : red")
            self.pushButton.setText("Turn off")
                # while self.worker1.ThreadActive:

        else:
            self.worker1.stop()
            if not self.worker1.ThreadActive:
                # print(self.worker1.ThreadActive)
                self.add_default_img()

            self.pushButton.setStyleSheet("background-color : green")
            self.pushButton.setText("Turn on")



class Worker1(QThread,LG_Remote):
    ImageUpdate = pyqtSignal(QImage)
    SignTxtUpdate = pyqtSignal(str)
    connection_response_txt = pyqtSignal(str)
    store_dict = pyqtSignal(dict)

    # client_obj = pyqtSignal(object)
    # INP_obj = pyqtSignal()
    def __init__(self,store,ip = None):
        super(Worker1,self).__init__()
        self.store = store
        self.Ip = ip
        # self.ThreadActive = False
        # self.qpixmap = QPixmap('re_plch.jpg')

    def Connect2TV(self,ip_add = None):
        try:
            # self.lineEdit.text():
            #     ip = '192.168.0.105'
            if len(self.store) == 0:
                # print(f'store len : {len(self.store)}')

                if ip_add is None:
                    self.connection_response_txt.emit("Searching LG WebOs tv connected to your local network.")
                    client = WebOSClient.discover(secure=True)[0]
                else:
                    client = WebOSClient(ip_add, secure=True)

                client.connect()
                for status in client.register(self.store):
                    if status == WebOSClient.PROMPTED:
                        self.connection_response_txt.emit("Please accept the connection on the TV!")
                        # print("Please accept the connection on the TV!")
                    elif status == WebOSClient.REGISTERED:
                        pass
                # print(self.store)

            elif len(self.store) != 0:
                # print(f'store len : {len(self.store)}')

                if ip_add is None:
                    self.connection_response_txt.emit("Searching LG WebOs tv connected to your local network.")
                    client = WebOSClient.discover(secure=True)[0]
                else:
                    client = WebOSClient(ip_add, secure=True)

                # client = WebOSClient.discover(secure=True)[0]

                # client = WebOSClient(ip, secure=True)

                client.connect()
                for status in client.register(self.store):
                    if status == WebOSClient.PROMPTED:
                        self.connection_response_txt.emit("Please accept the connection on the TV!")
                        # print("Please accept the connection on the TV!")
                    elif status == WebOSClient.REGISTERED:
                        pass

                    # print(self.store)

            # print("Tv is connected.\nNow you can control your tv with the handsigns.")
            self.connection_response_txt.emit("Tv is connected.\nNow you can control your tv with the hand signs.")

            # inp = InputControl(client)
            # Sys = SystemControl(client)
            # Sys.notify("LG AI remote is connected.")
            # self.client_obj.emit(client)
            return client,self.store
        except:
            self.connection_response_txt.emit("Your tv is switched off.\nBut hand signs could be detected.")
            # print("Your tv is switched off.\nBut handsigns could be detected.")

    def run(self):
        self.ThreadActive = True

        try:
            if self.Ip is not None:
                clnt,Store = self.Connect2TV(ip_add = self.Ip)
            else:
                clnt, Store = self.Connect2TV()

            # print(type(Store),Store)
            self.store_dict.emit(Store)
            inp = InputControl(clnt)

            Sys = SystemControl(clnt)
            Sys.notify("LG AI remote is connected.")
            signal = True

        except:
            signal = False
            # self.connection_response_txt.emit('No LG WebOs tv found on your local network.')
            # print('TV not found')


        Capture = cv.VideoCapture(0)

        # loading mediapipe
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                               max_num_hands=1,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5)

        # loading our handsign detection model
        HandSign_Classifier = HandsignClassifier()
        handsign_classifier_labels = ['Home','Back','OK','Up','Down','Left','Right','Unlocked','System_off','Locked']

        use_brect = True
        p_time = 0

        detection_time = 10
        waiting_time = 33

        # provide locking feature
        locking_flag = True

        # Counter variables
        count, count_1, count_2, count_3 = 0, 0, 0, 0

        count_4, count_5, count_6, count_7 = 0, 0, 0, 0

        count_8, count_9 = 0, 0

        while self.ThreadActive:

            # FPS Measurement
            n_time = time.time()
            fps = int(1 / (n_time - p_time))
            p_time = n_time
            # print(fps)

            mode = 0
            number = 0

            ret,frame = Capture.read()

            if not ret:
                break
            Image = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            FlippedImage = cv.flip(Image,1)
            debug_img = copy.deepcopy(FlippedImage)

            FlippedImage.flags.writeable = False
            res = hands.process(FlippedImage)
            FlippedImage.flags.writeable = True

            if res.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
                    brect = self.calc_bounding_rect(FlippedImage, hand_landmarks)

                    # landmark_list calculation
                    landmark_list = self.calc_landmark_list(FlippedImage, hand_landmarks)

                    # conversion to relative coordinates / normalized coordinates
                    pre_process_handlandmarks_list = self.pre_process_landmarks(landmark_list)
                    # print(pre_process_handlandmarks_list)
                    # Write the preprocessed dataset to the file
                    # self.logging_csv(number, mode, pre_process_handlandmarks_list)

                    # calling our model
                    hand_sign_label = HandSign_Classifier(pre_process_handlandmarks_list)

                    FlippedImage = self.draw_bounding_rect(use_brect, FlippedImage, brect)
                    FlippedImage = self.draw_landmarks(FlippedImage, landmark_list)

                    if hand_sign_label is not None:
                        FlippedImage = self.draw_info_text(FlippedImage, brect, handedness,handsign_classifier_labels[hand_sign_label])
                    #
                        self.SignTxtUpdate.emit(handsign_classifier_labels[hand_sign_label])

                    if signal:
                        if locking_flag:
                            if hand_sign_label == 0:  # Home signal
                                count += 1
                                if count == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.home()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")
                                        self.connection_response_txt.emit("Tv is Switched off.")

                                elif count == waiting_time:
                                    count = 0

                            if hand_sign_label == 1:  # back signal
                                count_1 += 1
                                if count_1 == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.back()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")


                                elif count_1 == waiting_time:
                                    count_1 = 0

                            if hand_sign_label == 2:  # Ok signal
                                count_2 += 1
                                if count_2 == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.ok()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")

                                elif count_2 == waiting_time:
                                    count_2 = 0

                            if hand_sign_label == 3:  # Up signal
                                count_3 += 1
                                if count_3 == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.up()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")


                                elif count_3 == waiting_time:
                                    count_3 = 0

                            if hand_sign_label == 4:  # Down signal
                                count_4 += 1
                                if count_4 == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.down()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")


                                elif count_4 == waiting_time:
                                    count_4 = 0

                            if hand_sign_label == 5:  # Left signal
                                count_5 += 1
                                if count_5 == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.left()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")


                                elif count_5 == waiting_time:
                                    count_5 = 0

                            if hand_sign_label == 6:  # Right signal
                                count_6 += 1
                                if count_6 == detection_time:
                                    try:
                                        inp.connect_input()
                                        inp.right()
                                        inp.disconnect_input()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")


                                elif count_6 == waiting_time:
                                    count_6 = 0

                            if hand_sign_label == 8:  # System_off signal
                                count_8 += 1
                                if count_8 == detection_time:
                                    try:
                                        Sys.power_off()
                                    except:
                                        self.connection_response_txt.emit("Tv is Switched off.")


                                elif count_8 == waiting_time:
                                    count_8 = 0

                        if hand_sign_label == 9:  # Locked signal
                            count_9 += 1
                            if count_9 == detection_time:
                                locking_flag = False
                                Sys.notify("LG AI remote is locked.")
                                # print("Locked")
                                self.connection_response_txt.emit("LG AI remote is locked.")
                            elif count_9 == waiting_time:
                                count_9 = 0

                        if hand_sign_label == 7:  # Unlocked signal
                            count_9 += 1
                            if count_9 == detection_time:
                                locking_flag = True
                                Sys.notify("LG AI remote is unlocked.")
                                # print("Unlocked")
                                self.connection_response_txt.emit("LG AI remote is unlocked.")

                            elif count_9 == waiting_time:
                                count_9 = 0

            FlippedImage = self.draw_info(FlippedImage, fps, mode, number)

            ConvertToQtFormat = QImage(FlippedImage.data,FlippedImage.shape[1],FlippedImage.shape[0],QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(960,540,Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)


    def stop(self):
        # img = cv.imread("re_plch.jpg")
        # convertoQtformat = QImage(img.data, img.shape[1], img.shape[0], QImage.Format_RGB888)
        # defalut_Pic = convertoQtformat.scaled(960, 540, Qt.KeepAspectRatio)
        # self.ImageUpdate.emit(defalut_Pic)
        self.ThreadActive = False
        self.quit()



if __name__ == '__main__':
    App = QApplication(sys.argv)
    widget = QStackedWidget()
    Root = MainWindow()
    # fet_screen = FeatureWindow()
    # widget.addWidget(Root)
    # widget.addWidget(fet_screen)
    # widget.setFixedWidth(755)
    # widget.setFixedHeight(810)
    # widget.show()
    Root.show()
    sys.exit(App.exec())


