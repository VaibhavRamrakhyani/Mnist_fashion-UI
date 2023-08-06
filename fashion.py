import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
# Load the trained CNN model
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = tf.keras.Sequential([
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(128, activation='relu'),
     tf.keras.layers.Dense(10)
 ])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                       'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def preprocess_image(image):
    # Resize and preprocess the image for the model
    resized_image = image.resize((28, 28))
    grayscale_image = resized_image.convert('L')
    image_array = np.array(grayscale_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    return image_array

def predict(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return np.argmax(prediction)


st.title("Fashion MNIST Classification")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
    prediction_class = predict(image)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    class_names_array=np.array(class_names)

    st.subheader("Prediction:")
    st.write(class_names[prediction_class])

    prediction_probabilities= model.predict(preprocess_image(image))[0]
    df = pd.DataFrame({'Class': class_names, 'Probability': prediction_probabilities})
    st.subheader("Prediction Probabilities:")
    st.bar_chart(df.set_index('Class'))