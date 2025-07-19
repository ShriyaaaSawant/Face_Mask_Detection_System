import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

INIT_LR = 1e-4
EPOCHS = 10
BS = 32
IMAGE_SIZE = (224, 224)

# Load and label data
data = []
labels = []
dataset_path = "Dataset"

for label in ["with_mask", "without_mask"]:
    path = os.path.join(dataset_path, label)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=IMAGE_SIZE)
        image = img_to_array(image)
        image = image / 255.0
        data.append(image)
        labels.append(label)

# Encode labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Split dataset
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.2, stratify=labels, random_state=42)

# Load base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))

# Add custom head
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base model layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile model
model.compile(loss="binary_crossentropy",
              optimizer=Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS),
              metrics=["accuracy"])

# Train model
model.fit(trainX, trainY, batch_size=BS,
          validation_data=(testX, testY), epochs=EPOCHS)

# Save trained model
model.save("mask_detector.h5")
print("✅ Model saved as 'mask_detector.h5'")
