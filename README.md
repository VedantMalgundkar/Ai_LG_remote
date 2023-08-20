
# LG Ai Remote

Ai remote for LG webos tv's. Mediapipe framework is used for recognising and retriving hand landmarks and futher on these landmarks tensorflow model is trained to detect handsigns.

## Acknowledgements

 - [reference](https://github.com/kinivi/hand-gesture-recognition-mediapipe.git)
 
 - [library for tv signal](https://github.com/supersaiyanmode/PyWebOSTV.git)


## Demo

#### Home

![Home](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/6962ad38-f512-477b-92da-2740c5b49267)

#### Back

![Back](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/9bc38c52-1b14-4736-8c4f-35c1f1595734)

#### OK

![OK](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/63d2fb82-235b-4a62-9990-a813c2f5821b)

#### Up

![Up](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/6fc0109a-3868-4200-861e-697cbd03f0ed)

#### Down

![Down](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/82a40807-85d0-479a-a5ff-47b91f9c3e88)

#### Left

![Left](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/ad09ea86-5a4e-40b4-9603-a50b9a9f2ca6)

#### Right

![Right](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/9425fbf8-df9c-42db-9f53-19c311a4e189)

#### Locked

![Locked](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/05f28ba8-f11c-4e28-a60d-f736864e5c1e)

#### Unlocked

![Unlocked](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/6f3a88d3-9138-4fe0-80be-e6a9ba92e57b)

#### System_off

![System_off](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/add9eb90-ce0b-46bd-9bf5-d620ec62a3d7)


## Run Locally

Clone the project

```bash
  git clone https://github.com/VedantMalgundkar/LG_Ai_remote.git
```


Install dependencies

```bash
  pip install -r requirements.txt
```

## Connect to tv

Run the Ai_remote.py file

Enter "y" if you want to connect to tv and then enter ip address of your tv.

![promt](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/89a70d80-36fd-405f-978e-bd7b31cafddb)

Accept this promt on the TV

![IMG_20230820_151113](https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/aa38c50c-0e49-4f95-8a3a-cc4e20d8c8e1)

you are good to go 👍


## Directory Structure

```
    │  Ai_remote.py
    │  accesskey.txt
    |  requirements.txt
    │  
    ├─model
        ├─Handsign_classifier
            │  handsign_classifier_datav4.csv
            |  handsign_classifier_labelsv4.csv
            │  handsign_classifierv4.hdf5
            │  handsign_classifierv4.tflite
            |_ handsign_classifier.py
           
```
#### Ai_remote.py
This is a sample program for detecting hand signs.

#### accesskey.txt
This file will contain access key which is provided by tv for connection.

#### handsign_classifier_datav4 & labelsv4
"data" file contain data retrived by mediapipe framework and "labels" has names for specific hand sign.

#### handsign_classifierv4.hdf5 & tflite
hdf5 is tensorflow keras model and tflite quantized version of that model.


## Data preprocessing

This function is reponsible for preprocessing the data.
````
def pre_process_landmarks(landmrk_list):
    temp_landmrk_list=copy.deepcopy(landmrk_list)
    base_x,base_y=0,0
    for id,landmrk in enumerate(temp_landmrk_list):
        if id == 0:
            base_x , base_y= landmrk[0],landmrk[1]

        temp_landmrk_list[id][0]= temp_landmrk_list[id][0]-base_x
        temp_landmrk_list[id][1]= temp_landmrk_list[id][1]-base_y

````
this much part of the code is just shifting all landmark points towords the origin.

example : https://www.geogebra.org/m/uwyhhvss

````
    # convert to a one dimensional list
    temp_landmrk_list = list(itertools.chain.from_iterable(temp_landmrk_list))

    # finding maximum value
    max_value=max(list(map(abs,temp_landmrk_list)))

    # normalization
    def normalize(n):
        return n/max_value

    temp_landmrk_list=list(map(normalize,temp_landmrk_list))

    return temp_landmrk_list

````
In this part normalize() function is scaling down all shifted points in the range of 0 to 1.

example : https://www.geogebra.org/m/qwj2v9sq

In above geogebra slide after clicking the destination button we can see all shifted points are scaled down in the range of 0 to 1.

We are scaling down image coordinates so that it becomes easy to train this preprocessed data to tensorflow keras model.


### This video will give us the idea about preprocessing step
https://github.com/VedantMalgundkar/LG_Ai_remote/assets/129035372/186fa1b3-6a95-48da-ada6-549df8ef93c2


## Training model

https://www.kaggle.com/code/vedantanilmalgundkar/handsign-classification

if you want to add your custom handsign data then

- Download "handsign-classification.ipynb" notebook from above link.

- Add your custom "handsign_classifier_datav4.csv" file to the directory.

- Change "no_of_classes" in the ipynb notebook, according to number of hand signs present in your the data.

- Use new "handsign_classifierv4.tflite" file instead of the old one.

- Change names in the "handsign_classifier_labelsv4.csv" according to your hand sign data.


