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

st.write("Importe ton image")

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    image_port = st.file_uploader("", type=["png", "jpg"])

    chiffre_pred = []

    if len(chiffre_pred) == 1:
        st.write(f'Le chiffre est prédit est {chiffre_pred[0]}')

with col2:
    if image_port:
        image_ported = Image.open(image_port)
        wpercent = (400/float(image_ported.size[0]))
        hsize = 400
        image_ported = image_ported.resize((400,hsize), Image.Resampling.LANCZOS)
        st.image(image_ported)

#bouton predict image
if st.button('Prédit ton image'):
    st.balloons()
    #conversion
    image_pred = np.array(image_ported.convert('L'))
    resized_img = cv2.resize(image_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255

    #affichage resultats en table
    st.write('Probabilité du résultat en pourcentage')
    y_pred_img = model.predict(resized_img)
    y_pred_img = (np.round(y_pred_img,3)*100).astype(int)
    pred_table = st.table(data=y_pred_img)
    chiffre_pred = pd.DataFrame(y_pred_img).T.idxmax()

    with col3:
        #affichage traitement
        st.write("Ce que voit l'IA")
        fig1, ax1 = plt.subplots()
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax1.imshow(resized_img[0],cmap='gray')
        st.pyplot(fig=fig1)
