# for video capture and image manipulation
import cv2 as cv

# for processing
import numpy as np
import csv
import copy
import itertools
import os

# importing models
import mediapipe as mp
from model import HandsignClassifier

# for controlling LG webos tv
from pywebostv.connection import *
from pywebostv.controls import *


def select_mode(key,mode):
    number = -1
    if 48 <= key <= 57:
        number= key - 48
    if key == 110: # n - for normal screen
        mode = 0
    if key == 104: # h - for saving hand sign
        mode= 1

    return number , mode

def calc_landmark_list(img,landmrks):
    img_width, img_height = img.shape[1], img.shape[0]

    landmark_list=[]

    for id,landmark in enumerate(landmrks.landmark):
        landmark_x = min(int(landmark.x * img_width), img_width-1)
        landmark_y = min(int(landmark.y * img_height),img_height-1)


        landmark_list.append([landmark_x,landmark_y])

    return landmark_list

def calc_bounding_rect(img, landmrks):
    landmark_list = calc_landmark_list(img, landmrks)
    landmrk_array = np.array(landmark_list).reshape(21, 2)

    x, y, w, h = cv.boundingRect(landmrk_array)  # boundingRect require numpy array

    return [x, y, x + w, y + h]

def pre_process_landmarks(landmrk_list):
    temp_landmrk_list=copy.deepcopy(landmrk_list)
    base_x,base_y=0,0
    for id,landmrk in enumerate(temp_landmrk_list):
        if id == 0:
            base_x , base_y= landmrk[0],landmrk[1]

        temp_landmrk_list[id][0]= temp_landmrk_list[id][0]-base_x
        temp_landmrk_list[id][1]= temp_landmrk_list[id][1]-base_y

    # convert to a one dimensional list
    temp_landmrk_list = list(itertools.chain.from_iterable(temp_landmrk_list))

    # finding maximum value
    max_value=max(list(map(abs,temp_landmrk_list)))

    # normalization
    def normalize(n):
        return n/max_value

    temp_landmrk_list=list(map(normalize,temp_landmrk_list))

    return temp_landmrk_list

def logging_csv(number,mode,landmark_lst):
    if mode == 0:
        pass
    if mode == 1 and (0<=number<=9):
        with open("model/Handsign_classifier/handsign_classifier_datav4.csv", 'a', newline='') as f:
            write = csv.writer(f)
            write.writerow([number,*landmark_lst])

    return

def draw_landmarks(image, landmark_point):
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

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

def draw_info(image, fps, mode, number):
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

def draw_info_text(image, brect, handedness, hand_sign_text):

    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image



def main():

    connect_button = input('Want to connect LG tv ? type "y" or leave it empty : ')

    if connect_button=='y':
        ip = input('Enter ip address of your tv : ')
    else:
        print('But still you can see the handsign predictions.')

    signal = False

    try:
        if connect_button == 'y':
            store = {}
            if len(store) == 0 and os.path.getsize('accesskey.txt')!= 0:

                with open('accesskey.txt') as f:
                    auth_key = json.load(f)
                    store['client_key'] = auth_key['client_key']
                    f.close()

                client = WebOSClient(ip, secure=True)
                client.connect()
                for status in client.register(store):
                    if status == WebOSClient.PROMPTED:
                        print("Please accept the connection on the TV!")
                    elif status == WebOSClient.REGISTERED:
                        pass

                # print("Tv is connected.\nNow you can control your tv with the handsigns.")


            elif len(store) == 0 and os.path.getsize('accesskey.txt')== 0:

                client = WebOSClient(ip, secure=True)

                client.connect()
                for status in client.register(store):
                    if status == WebOSClient.PROMPTED:
                        print("Please accept the connection on the TV!")
                    elif status == WebOSClient.REGISTERED:
                        pass

                with open('accesskey.txt','w') as f:
                    f.write(json.dumps(store))
                    f.close()

            print("Tv is connected.\nNow you can control your tv with the handsigns.")


            inp = InputControl(client)
            sys = SystemControl(client)
            sys.notify("LG AI remote is connected.")

            signal = True
    except:
        print("Your tv is switched off.\nBut still you can see the handsign predictions.")


    use_brect=True

    #Camera preparation
    cap= cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,960)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT,540)

    # Mediapipe model loading
    mp_hands= mp.solutions.hands
    hands= mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

    # loading our handsign detection model
    HandSign_Classifier = HandsignClassifier()

    # Reading labels
    with open("model/Handsign_classifier/handsign_classifier_labelsv4.csv", encoding='utf-8-sig') as f:
        lines=csv.reader(f)
        handsign_classifier_labels=[i[0] for i in lines]



    p_time= 0

    # Adjust these variables to control speed of input signal.
    detection_time = 10
    waiting_time = 33

    mode = 0

    # provide locking feature
    locking_flag= True

    # Counter variables
    count,count_1,count_2,count_3 = 0,0,0,0

    count_4,count_5,count_6,count_7 =0,0,0,0

    count_8,count_9 = 0,0

    while True:

        # FPS Measurement
        n_time=time.time()
        fps = int(1/(n_time-p_time))
        p_time = n_time

        #Process key (Esc : end)
        key = cv.waitKey(10)
        if key == 27:
            break

        number,mode = select_mode(key,mode)

        # Camera capture
        result, img = cap.read()
        if not result:
            break
        img = cv.flip(img,1)
        debug_img= copy.deepcopy(img)

        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

        img.flags.writeable=False
        res = hands.process(img)
        img.flags.writeable=True

        if res.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(res.multi_hand_landmarks,res.multi_handedness):

                # bounding box calculation
                brect = calc_bounding_rect(debug_img,hand_landmarks)

                # landmark_list calculation
                landmark_list =calc_landmark_list(debug_img,hand_landmarks)


                # conversion to relative coordinates / normalized coordinates
                pre_process_handlandmarks_list= pre_process_landmarks(landmark_list)


                # Write the preprocessed dataset to the file
                logging_csv(number, mode, pre_process_handlandmarks_list)

                # calling our model
                hand_sign_label=HandSign_Classifier(pre_process_handlandmarks_list)

                if signal:
                    if locking_flag:
                        if hand_sign_label == 0:# Home signal
                            count+=1
                            if count == detection_time:
                                try:
                                    inp.connect_input()
                                    inp.home()
                                    inp.disconnect_input()
                                except:
                                    print("Tv is Switched off.")

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
                                    print("Tv is Switched off.")


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
                                    print("Tv is Switched off.")

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
                                    print("Tv is Switched off.")


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
                                    print("Tv is Switched off.")


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
                                    print("Tv is Switched off.")


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
                                    print("Tv is Switched off.")


                            elif count_6 == waiting_time:
                                count_6 = 0

                        if hand_sign_label == 8:  # System_off signal
                            count_8 += 1
                            if count_8 == detection_time:
                                try:
                                    sys.power_off()
                                except:
                                    print("Tv is Switched off.")


                            elif count_8 == waiting_time:
                                count_8 = 0


                    if hand_sign_label == 9:  # Locked signal
                        count_9 += 1
                        if count_9 == detection_time:
                            locking_flag = False
                            sys.notify("LG AI remote is locked.")
                            print("Locked")

                        elif count_9 == waiting_time:
                            count_9 = 0

                    if hand_sign_label == 7:  # Unlocked signal
                        count_9 += 1
                        if count_9 == detection_time:
                            locking_flag = True
                            sys.notify("LG AI remote is unlocked.")
                            print("Unlocked")

                        elif count_9 == waiting_time:
                            count_9 = 0


                # Drawing part

                debug_img = draw_bounding_rect(use_brect, debug_img, brect)
                debug_img = draw_landmarks(debug_img, landmark_list)

                if hand_sign_label is not None:
                    debug_img = draw_info_text(debug_img,brect,handedness,handsign_classifier_labels[hand_sign_label])


        debug_img = draw_info(debug_img, fps, mode, number)

        cv.imshow('Handsign Recognition',debug_img)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()



















