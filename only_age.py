from PIL import Image, ImageFont, ImageDraw
# from luggage.yolo_bd import luggage
import cv2, helper, numpy as np, requests
import tensorflow as tf
from keras.models import model_from_json

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
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
colors = ["blueviolet", "brown", "pink", "orange", "green", "blue"]
def ageModel():
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
    labels = 101
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

age_model = ageModel() 
def age(face):
    prediction = age_model.predict(face, verbose=0)
    apparent_age = int(findApparentAge(prediction))
    return apparent_age
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
        with tf.device('/CPU:0'):
            draw.rectangle([(linep[0], linep[1]), (linep[2], linep[3])], outline=colors[color_pic], width=5)
            draw.text((linep[0], linep[3]), f"{age_result}tahun", "white", font=font)
            draw.text((linep[0], (linep[3]+46)), f"Foto Ke-{i}", "white", font=font)
    return np.array(image)
with tf.device('/CPU:0'):
    font = ImageFont.truetype("./FiraCode-Regular.ttf", 22)
    cap = cv2.VideoCapture(0)
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