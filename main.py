import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the directories
cancer_images = (
    "cancer/"  # This directory contains two subdirectories: 'cancer' and 'non-cancer'
)

# Use ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

# Load the images using flow_from_directory
images = datagen.flow_from_directory(
    cancer_images,
    target_size=(150, 150),  # Adjust the target size according to your image size
    batch_size=32,
    class_mode="binary",
    subset="training",
    seed=123,
)

validation_images = datagen.flow_from_directory(
    cancer_images,
    target_size=(150, 150),  # Adjust the target size according to your image size
    batch_size=32,
    class_mode="binary",
    subset="validation",
    seed=123,
)

# Define the model
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
X_test, y_test = validation_images.next()
# Concatenate the data
X = np.concatenate((images[0][0], validation_images[0][0]), axis=0)
y = np.concatenate((images[0][1], validation_images[0][1]), axis=0)

# Check if the generators have data
if len(images[0][0]) > 0 and len(validation_images[0][0]) > 0:
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    print("No data available in the generators.")


# Normalize the image data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)


# Now you can use the make_prediction function as before
def make_prediction(image):
    processed_image = preprocess_image(image)
    reshaped_image = np.reshape(
        processed_image,
        (
            1,
            processed_image.shape[0],
            processed_image.shape[1],
            processed_image.shape[2],
        ),
    )
    prediction = model.predict(reshaped_image)
    return prediction
