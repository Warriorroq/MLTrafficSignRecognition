import cv2
import numpy as np

import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import pickle


path = "testing_multiclass.txt"
filenames = open(path).read().strip().split("\n")
imagePaths = []

for f in filenames:
    imagePaths.append(f)

modelName = input("model id:")
model = load_model(f"model-{modelName}.keras")
lb = pickle.loads(open(f"lb-{modelName}.pickle", "rb").read())


#os.system('pip install playsound')
#os.system('pip install gTTS')
#os.system('pip install pyttsx3')

for imagePath in imagePaths[:15]:

    # loading input image
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # predicting bbox and label
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    # finding class label with highest pred. probability
    i = np.argmax(labelPreds, axis=1)
    label = lb.classes_[i][0]

    # loading the image in OpenCV format
    image = cv2.imread(imagePath)
    # image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]

    # scaling pred. bbox coords according to image dims
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)

    # drawing bbox and label on image
    y = startY - 10 if startY - 10 > 10 else startY + 10

    cv2.putText(image,
                label,
                (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.rectangle(image,
                  (startX, startY),
                  (endX, endY),
                  (0, 255, 0),
                  2)


    # showing the output image
    imgplot = plt.imshow(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8'))
    plt.show()
    print(label)