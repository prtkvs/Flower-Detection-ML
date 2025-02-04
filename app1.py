import os
import random
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np
from PIL import Image

st.header('Flower Classification CNN Model')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

model = load_model('D:\CODING PRTK\Flower Recognition\Flower_Recog_Model.h5')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)
    
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    predicted_label = flower_names[np.argmax(result)]
    outcome = f'The Image belongs to {predicted_label} with a score of {np.max(result)*100:.2f}%'
    
    return predicted_label, outcome

def get_random_images(folder, num_images=10):
    image_paths = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith(('png', 'jpg', 'jpeg'))]
    return random.sample(image_paths, min(num_images, len(image_paths)))

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
    image_path = os.path.join('upload', uploaded_file.name)
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)
    predicted_label, result_text = classify_images(image_path)
    st.markdown(result_text)
    
    if predicted_label in flower_names:
        st.subheader(f'Random {predicted_label} Images')
        flower_folder = rf'D:\CODING PRTK\Flower Recognition\Images\{predicted_label.lower()}'  # Folder containing sunflower images
        if os.path.exists(flower_folder):
            flower_images = get_random_images(flower_folder)
            for img_path in flower_images:
                st.image(Image.open(img_path), width=150)
        else:
            st.warning('related images not found!')
    
    st.markdown(result_text) 


