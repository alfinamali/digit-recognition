import keras as kr
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

# Image dimensions
img_rows, img_cols = 28, 28

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data
if K.image_data_format() == "channels_first":
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# Normalize
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# One-hot encode labels
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Augmentasi data
datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
)
datagen.fit(x_train)

# Build model
model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        input_shape=input_shape,
    )
)
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation="softmax"))

# Compile
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train
model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=15,
    validation_data=(x_test, y_test),
    verbose=1,
)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("\n\nLoss: %6.4f\tAccuracy: %6.4f" % (loss, accuracy))

# Predict a sample
prediction = np.around(model.predict(np.expand_dims(x_test[0], axis=0))).astype(int)[0]
print("Actual: %s\tEstimated: %s" % (y_test[0].astype(int), prediction))

# Save model
model_json = model.to_json()
with open("model/mnistModel.json", "w") as json_file:
    json_file.write(model_json)
model.save("model/mnistModel.keras")
