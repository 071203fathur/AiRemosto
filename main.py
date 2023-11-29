from PIL import Image, ImageFont, ImageDraw
# from luggage.yolo_bd import luggage
import cv2, helper, numpy as np, requests
import tensorflow as tf
from keras.models import model_from_json
from datetime import datetime
import streamlink

# from age import VGGFace

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3500)])
  except RuntimeError as e:
    print(e)
tf_version = int(tf.__version__.split(".", maxsplit=1)[0])

if tf_version == 1:
    from keras.models import Model, Sequential
    from keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Dense,
    )
else:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        Convolution2D,
        ZeroPadding2D,
        MaxPooling2D,
        Flatten,
        Dropout,
        Activation,
        Dense,
    )
race_labels = ["Asian", "Indian", "Negroid", "Caucasian ", "Middle Eastern", "Latino"]
gender_labels = ["prempuan", "laki-laki"]

# VGGmodel = VGGFace.baseModel()
def vggBaseModel():
    # --------------------------
    
    
    labels_gender = 2
    #classes = 6
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation="relu"))
    model.add(Dropout(0.5))
    
    return model

# vgg_base_model =  

def raceModel():
    labels = 6
    model = vggBaseModel()
    model.add(Convolution2D(labels, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))
    result_model = Model(inputs=model.input, outputs=model.output)
    result_model.load_weights("race/race_model_single_batch.h5")
    return result_model

def genderModel():
    labels = 2
    model = vggBaseModel()
    model.add(Convolution2D(labels, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))
    result_model = Model(inputs=model.input, outputs=model.output)
    result_model.load_weights("gender/gender_model_weights.h5")
    return result_model

def age_loadModel():
    # --------------------------
    labels = 101
    model = vggBaseModel()
    model.add(Convolution2D(labels, (1, 1)))
    model.add(Flatten())
    model.add(Activation("softmax"))
    result_model = Model(inputs=model.input, outputs=model.output)
    result_model.load_weights("age/age_model_weights.h5")
    return result_model

def findApparentAge(age_predictions):
    output_indexes = np.array(list(range(0, 101)))
    apparent_age = np.sum(age_predictions * output_indexes)
    return apparent_age

#Load Models that base on VGG
gender_model = genderModel()
# race_model= raceModel()
age_model = age_loadModel()
# Emotions stuff
emotion_dict = ["marah", "risih", "takut", "senyum", "netral", "sedih", "terkejut"]
emotion_prediction = [[0, 0, 0, 0, 0, 0, 0]]
json_file = open('expression/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("expression/emotion_model.h5")
colors = ["blueviolet", "brown", "pink", "orange", "green", "blue"]

# def expression(face):
#     gray_frame = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(gray_frame, (48, 48)), -1), 0)
#     emotion_prediction = emotion_model.predict(cropped_img, verbose=0)
#     maxindex = int(np.argmax(emotion_prediction))
#     return emotion_dict[maxindex]

def age(face):
    prediction = age_model.predict(face, verbose=0)
    apparent_age = int(findApparentAge(prediction))
    return apparent_age

def gender(face):
    gender_result = {}
    gender_predictions_real =  gender_model.predict(face, verbose=0)
    gender_predictions = gender_predictions_real[0, :]
    for i, gender_label in enumerate(gender_labels):
        gender_prediction = 100 * gender_predictions[i]
        gender_result[gender_label] = gender_prediction
    dominant_gender = gender_labels[np.argmax(gender_predictions)]
    return dominant_gender, gender_predictions_real

# def race(face):
#     race_predictions_real =  race_model.predict(face, verbose=0)
#     race_predictions = race_predictions_real[0, :]
#     sum_of_predictions = race_predictions.sum()
#     race_prediction_min = 0
#     dominant_race = ""
#     for i, race_label in enumerate(race_labels):
#         race_prediction = 100 * race_predictions[i] / sum_of_predictions
#         if race_prediction_min < race_prediction:
#             race_prediction_min = race_prediction
#             dominant_race =  race_label
#     return dominant_race,race_predictions_real
def get_youtube_stream(url):
    try:
        # Create a Streamlink session
        session = streamlink.Streamlink()

        # Retrieve streams
        streams = session.streams(url)

        if streams:
            # Print available stream qualities
            print("Available stream qualities:", list(streams.keys()))

            # Choose the desired quality (you can customize this based on your needs)
            chosen_quality = 'best'
            best_stream = streams[chosen_quality]
            
            print(f"Chosen Quality: {chosen_quality}")
            print(f"Stream URL: {best_stream.url}")
            
            return best_stream.url
        else:
            print("No streams found.")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None




# Replace 'your_youtube_url' with the actual URL of the live YouTube stream

def faces_model(frame):
    image = Image.fromarray(frame)
    landmarks = face_detector(image)
    draw = ImageDraw.Draw(image)
    pria, wanita, anak, remaja, dewasa, lansia = [0]*6 
    for i, landmark in enumerate(landmarks):
        with tf.device('/CPU:0'):
            linep = helper.crop_face(image, landmark, expand=.6)
            face_img = helper.align_and_crop_face(image, linep, 224)
            linep = linep[0]
            color_pic = i % len(colors)
            face = np.asarray(face_img)
            face = face.astype("float32")
            mean, std = face.mean(), face.std()
            face = (face-mean)/std
            face = np.expand_dims(face_img, axis=0)
        age_result= age(face)
        gender_result, gender_pr= gender(face)
        with tf.device('/CPU:0'):
        # race_result, race_pr= race(face)
        # expression_result = expression(cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR))
            draw.rectangle([(linep[0], linep[1]), (linep[2], linep[3])], outline=colors[color_pic], width=5)
            draw.text((linep[0], (linep[1]-38)), gender_result, "white", font=font)
            # draw.text((linep[0], (linep[1]-23)), f"{expression_result}", "white", font=font)
            draw.text((linep[0], linep[3]), f"{age_result}tahun", "white", font=font)
            # draw.text((linep[0], (linep[3]anak+23)), f"{race_result}", "white", font=font)
            draw.text((linep[0], (linep[3]+46)), f"Foto Ke-{i}", "white", font=font)
            now =  datetime.now().timestamp()
            if gender_result == "laki-laki":
                pria += 1
            else:
                wanita += 1
            if int(age_result) <= 12:
                anak += 1
            if (int(age_result) <= 25) and (int(age_result) > 12) :
                remaja += 1
            if int(age_result) <= 40 and (int(age_result) > 25):
                dewasa += 1
            if int(age_result) > 40:
                lansia += 1
    with tf.device('/CPU:0'):
        now = int(datetime.now().timestamp())
        values={
                "gender": {
                    "created_at" : now,
                    "pria": pria,
                    "wanita" : wanita 
                },
                "age" : {
                    "created_at" : now,
                    "anak" : anak,
                    "remaja" : remaja,
                    "dewasa" : dewasa,
                    "lansia" : lansia
                },
                "race" : {
                    "created_at" : now,
                    "negroid" : 0,
                    "east_asian" : 0,
                    "indian" : 0,
                    "latin" : 0,
                    "middle_eastern" : 0,
                    "south_east_asian" : 0,
                    "kauskasia" : 0,
                },
                "luggage" : {
                    "created_at" : now,
                    "manusia" : 0,
                    "besar" : 0,
                    "sedang" : 0,
                    "kecil": 0,
                },
                "expression" : {
                    "created_at" : now,
                    "marah" : 0,
                    "risih" : 0,
                    "takut" : 0,
                    "senyum" : 0,
                    "netral" : 0,
                    "sedih" : 0,
                    "terkejut" : 0
                }
            }
        draw.text((0, 0), f"Data: {values}", "white", font=font)
        t = requests.post("http://localhost:3000/api/camera", json=values)
        print(f"requested: {values}")
        
        print(f"response: {t.json()}")
    return np.array(image)
with tf.device('/CPU:0'):
    font = ImageFont.truetype("./FiraCode-Regular.ttf", 22)
    youtube_url = 'https://www.youtube.com/watch?v=x9ABPCKYZgs'
    stream_url = get_youtube_stream(youtube_url)
    cap = cv2.VideoCapture(stream_url)
    face_detector = helper.get_dlib_face_detector()
while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_result = faces_model(frame_rgb)
    with tf.device('/CPU:0'):
        frame_result = cv2.cvtColor(frame_result, cv2.COLOR_RGB2BGR)
        # frame_result = luggage(frame_result)
        cv2.imshow("REMOSTO",frame_result)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break