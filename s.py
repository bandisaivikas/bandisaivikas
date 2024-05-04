import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

print("Welcome to the NeuralNine (c) Handwritten Digits Recognition v0.1")

# Function to handle model prediction
def predict_digit(image_path):
    try:
        img = cv2.imread(image_path)[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        digit_label.config(text="TThe selected number is {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except Exception as e:
        digit_label.config(text="Error predicting image! " + str(e))

# Function to handle file selection
def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        predict_digit(file_path)

# Decide if to load an existing model or to train a new one
train_new_model = True

if train_new_model:
    # Loading the MNIST data set with samples and splitting it
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalizing the data (making length = 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

    # Create a neural network model
    # Add one flattened input layer for the pixels
    # Add two dense hidden layers
    # Add one dense output layer for the 10 digits
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

    # Compiling and optimizing model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training the model
    model.fit(X_train, y_train, epochs=3)

    # Evaluating the model
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Saving the model
    model.save('handwritten_digits.h5')

else:
    # Load the model
    model = tf.keras.models.load_model('handwritten_digits.model')

# Create a Tkinter window
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# Create a button to select an image
select_button = tk.Button(root, text="Select Image", command=select_file)
select_button.pack(pady=10)

# Label to display prediction result
digit_label = tk.Label(root, text="")
digit_label.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
