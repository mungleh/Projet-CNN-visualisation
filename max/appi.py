import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#model
model = tf.keras.models.load_model('modail.h5')

st.set_page_config(
    layout='wide',
)


col1, col2 = st.columns(2, gap="large")


with col1:

    sdbtt1 = st.title("Déssine ou Importe une image")

    stroke_width = st.slider("Taile du dessinage", 1, 25, 10)
    realtime_update = st.checkbox("Actualisation en live", True)
    #canvas
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
        #affichage traitement
        st.write("Ce que voit l'IA")
        fig, ax = plt.subplots(figsize=(1,1))
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.imshow(resized[0],cmap='gray')
        st.pyplot(fig=fig)


with col2:

    image_port = st.file_uploader("IMPOOOOOOOOOOOOOOOOOORTE", type=["png", "jpg"])

    if image_port == None:
        st.text("")
    else:
        image_ported = Image.open(image_port)
        wpercent = (400/float(image_ported.size[0]))
        hsize = 400
        image_ported = image_ported.resize((400,hsize), Image.Resampling.LANCZOS)
        st.image(image_ported)

    #bouton predict image
    if st.button('Prédit ton image'):
        st.snow()
        #conversion
        image_pred = np.array(image_ported.convert('L'))
        resized_img = cv2.resize(image_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255
        #affichage resultats en table
        st.write('Probabilité du résultat en pourcentage')
        y_pred_img = model.predict(resized_img)
        y_pred_img = (np.round(y_pred_img,3)*100).astype(int)
        st.table(data=y_pred_img)
        #affichage traitement
        st.write("Ce que voit l'IA")
        fig1, ax1 = plt.subplots(figsize=(1,1))
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax1.imshow(resized_img[0],cmap='gray')
        st.pyplot(fig=fig1)
