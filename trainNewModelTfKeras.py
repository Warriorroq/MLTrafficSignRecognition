# !pip install imutils
import sys
import os
import cv2
import datetime
import string
# import imutils
import numpy as np
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import pickle


data = []
labels = []
bboxes = []
imagePaths = []

# classes = ["trafficlight", "speedlimit", "crosswalk", "stop"]

annot_dir  = "./data/annotations"
images_dir = "./data/images"

for filename in os.listdir(annot_dir):
    f = os.path.join(annot_dir, filename)
    tree = ET.parse(f)
    root = tree.getroot()

    w = int(root.find('.//size/width').text)
    h = int(root.find('.//size/height').text)

    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text) / w
        ymin = int(box.find('ymin').text) / h
        xmax = int(box.find('xmax').text) / w
        ymax = int(box.find('ymax').text) / h

    label = root.find('.//object/name').text

    imname = root.find('.//filename').text
    impath = os.path.join(images_dir, imname)
    image = load_img(impath, target_size=(224, 224))
    image = img_to_array(image)

    data.append(image)
    labels.append(label)
    bboxes.append((xmin, ymin, xmax, ymax))
    imagePaths.append(impath)

# normalize -> from [0-255] to [0-1]
data = np.array(data, dtype="float32") / 255.0

# convert to np arrays
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

# one-hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# test-train split 20%,80%

split = train_test_split(data,
                         labels,
                         bboxes,
                         imagePaths,
                         test_size=0.20,
                         random_state=12)


(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths,  testPaths)  = split[6:]

# saving test files for later use
with open("testing_multiclass.txt", "w") as f:
    f.write("\n".join(testPaths))

#The Neural Net : Architecture
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50
typeOfModel = input("what type of model do you want? [VGG16, MobileNetV2, InceptionV3, ResNet50]").strip()
model = 0

if typeOfModel == "MobileNetV2":
    model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
elif typeOfModel == "InceptionV3":
    model = InceptionV3(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
elif typeOfModel == "ResNet50":
    model = ResNet50(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
else:
    model = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# freeze training any of the layers of VGGNet
model.trainable = False

# max-pooling is output of VGG, flattening it further
flatten = model.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)
# 4 neurons correspond to 4 co-ords in output bbox

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(
    inputs=model.input,
    outputs=(bboxHead, softmaxHead))

INIT_LR = 1e-4
NUM_EPOCHS = int(input("Amount of Epochs:"))
BATCH_SIZE = 16
isEarlyStopping = True if input("Apply early stopping? t/f").__contains__('t') else False
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}

testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

opt = Adam(INIT_LR)

model.compile(loss=losses,
              optimizer=opt,
              metrics=["accuracy", 'accuracy'],
              loss_weights=lossWeights)

print(model.summary())

#Training
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping, lr] if isEarlyStopping else [lr],
    verbose=1)

modelName = "".join(random.choices(string.ascii_uppercase + string.digits, k=9))
model.save(f'model-{modelName}.keras')


f = open(f"lb-{modelName}.pickle", "wb")
f.write(pickle.dumps(lb))
f.close()














def plot_sample_images(images, labels, lb, grid_size=(4, 5)):
    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    axes = axes.flatten()
    for img, label, ax in zip(images, labels, axes):
        ax.imshow(img)
        ax.set_title(lb.classes_[np.argmax(label)])
        ax.axis('off')
    plt.tight_layout()
    pltPath = os.path.sep.join([f"sample_images_grid-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()

def plot_misclassified_images(images, true_labels, pred_labels, lb, grid_size=(4, 4)):
    fig, axes = plt.subplots(*grid_size, figsize=(12, 12))
    axes = axes.flatten()
    for img, true_label, pred_label, ax in zip(images, true_labels, pred_labels, axes):
        ax.imshow(img)
        ax.set_title(f'True: {lb.classes_[true_label]}\nPred: {lb.classes_[pred_label]}')
        ax.axis('off')
    plt.tight_layout()
    pltPath = os.path.sep.join([f"misclassified_images-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()

#plotting
def plotting():
    N = np.arange(0, len(H.history['loss']))
    plt.style.use("ggplot")
    (fig, ax) = plt.subplots(3, 1, figsize=(17, 10))

    # plotting the loss for training and validation data

    for (i, l) in enumerate(["loss", "class_label_accuracy", "bounding_box_accuracy"]):
        title = "Loss for {}".format(l) if l != "loss" else "Total loss"
        ax[i].set_title(title)
        ax[i].set_xlabel("Epoch #")
        ax[i].set_ylabel("Loss")
        ax[i].plot(N, H.history[l], label=l)
        ax[i].plot(N, H.history["val_" + l], label="val_" + l)
        ax[i].legend()
    
    plotPath = os.path.sep.join([f"data-{modelName}.png"])
    plt.savefig(plotPath)
    plt.show()
    # create a new figure for the accuracies
    plt.style.use("ggplot")
    plt.figure(figsize=(17, 10))

    plt.plot(N, H.history["class_label_accuracy"],
         label="class_label_train_acc")
    plt.plot(N, H.history["val_class_label_accuracy"],
         label="val_class_label_acc")

    plt.title("Class Label Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    # save the accuracies plot
    plotPath = os.path.sep.join([f"acc_label-{modelName}.png"])
    plt.savefig(plotPath)

    # create a new figure for the accuracies
    plt.style.use("ggplot")
    plt.figure(figsize=(17, 10))

    plt.plot(N, H.history["bounding_box_accuracy"],
         label="bounding_box_train_acc")
    plt.plot(N, H.history["val_bounding_box_accuracy"],
         label="val_bounding_box_acc")

    plt.title("Bounding Box Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")

    # save the accuracies plot
    plotPath = os.path.sep.join([f"acc_bbox-{modelName}.png"])
    plt.savefig(plotPath)

    import seaborn as sns

    # Class distribution
    class_counts = np.sum(trainLabels, axis=0)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=lb.classes_, y=class_counts)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    pltPath = os.path.sep.join([f"class_distribution-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()

    # Plot sample images
    plot_sample_images(trainImages[:20], trainLabels[:20], lb)



    plt.figure(figsize=(10, 6))
    plt.plot(H.history['loss'], label='Train Loss')
    plt.plot(H.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    pltPath = os.path.sep.join([f"training_validation_loss-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()


    plt.figure(figsize=(10, 6))
    plt.plot(H.history['class_label_accuracy'], label='Train Accuracy')
    plt.plot(H.history['val_class_label_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    pltPath = os.path.sep.join([f"training_validation_accuracy-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()


    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Predict the test set results
    y_pred = model.predict(testImages)[1]
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(testLabels, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=lb.classes_, yticklabels=lb.classes_)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    pltPath = os.path.sep.join([f"confusion_matrix-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()


    from sklearn.metrics import precision_recall_curve

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(lb.classes_):
        precision, recall, _ = precision_recall_curve(testLabels[:, i], y_pred[:, i])
        plt.plot(recall, precision, label=class_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pltPath = os.path.sep.join([f"precision_recall_curve-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()


    from sklearn.metrics import roc_curve, auc

    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(lb.classes_):
        fpr, tpr, _ = roc_curve(testLabels[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    pltPath = os.path.sep.join([f"roc_curve-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()


    misclassified_indices = np.where(y_pred_classes != y_true)[0]
    misclassified_images = testImages[misclassified_indices]
    misclassified_labels = y_pred_classes[misclassified_indices]
    true_labels = y_true[misclassified_indices]

    # Plot misclassified images
    plot_misclassified_images(misclassified_images[:16], true_labels[:16], misclassified_labels[:16], lb)


    from tensorflow.keras.models import Model
    import numpy as np

    # Define a new model that outputs the activation maps
    layer_outputs = [layer.output for layer in model.layers[1:12]] # Adjust the layers accordingly
    activation_model = Model(inputs=model.input, outputs=layer_outputs)

    # Get activation maps for a sample image
    sample_image = testImages[0].reshape(1, 224, 224, 3)
    activations = activation_model.predict(sample_image)

    # Plot activation maps
    layer_names = [layer.name for layer in model.layers[1:12]]
    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_features = layer_activation.shape[-1]
        size = layer_activation.shape[1]
        n_cols = n_features // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    pltPath = os.path.sep.join([f"activation_maps-{modelName}-{layer_name}.png"])
    plt.savefig(pltPath)
    plt.show()


    from sklearn.manifold import TSNE
    import seaborn as sns

    # Extract features using the VGG16 base model
    feature_model = Model(inputs=model.input, outputs=flatten)
    features = feature_model.predict(testImages)

    # Reduce dimensions with T-SNE
    tsne = TSNE(n_components=2)
    tsne_features = tsne.fit_transform(features)

    # Plot T-SNE
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], hue=np.argmax(testLabels, axis=1), palette='bright', legend=None)
    plt.title('T-SNE Plot')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    pltPath = os.path.sep.join([f"tsne_plot-{modelName}.png"])
    plt.savefig(pltPath)
    plt.show()


    from sklearn.metrics import precision_score, recall_score, f1_score

    # Calculate precision, recall, and f1-score for each class
    precision = precision_score(y_true, y_pred_classes, average=None)
    recall = recall_score(y_true, y_pred_classes, average=None)
    f1 = f1_score(y_true, y_pred_classes, average=None)
    print(precision, recall, f1)

if input("Display more data? t/f").__contains__('t'):
    plotting()