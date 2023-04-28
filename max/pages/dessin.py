import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit_nested_layout
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#model
model = tf.keras.models.load_model('modail.h5')

st.set_page_config(
    layout="wide"
)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    stroke_width = st.slider("Taile du dessinage", 1, 25, 10)
    realtime_update = st.checkbox("Actualisation en live", True)

with col2:
    #canvas
    st.write("Déssine")
    canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color='black',
    background_color="white",
    update_streamlit=realtime_update,
    height=300,
    width=300,
    drawing_mode='freedraw',
    display_toolbar=True,
    key="canvas"
    )

if canvas_result.image_data is not None:
    dessin = canvas_result.image_data

#bouton predict dessin
if st.button('Prédit ton dessin'):
    st.snow()
    #conversion
    dessin_pred = np.array(Image.fromarray(dessin).convert('L'))
    resized = cv2.resize(dessin_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255

    #affichage resultats en table
    st.write('Probabilité du résultat en pourcentage')
    y_pred = model.predict(resized)
    st.table(data=(np.round(y_pred,3)*100).astype(int))

    with col3:
        #affichage traitement
        st.write("Ce que voit l'IA")
        fig, ax = plt.subplots(figsize=(1,1))
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.imshow(resized[0],cmap='gray')
        st.pyplot(fig=fig)
