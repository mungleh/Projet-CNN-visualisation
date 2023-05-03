import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit_nested_layout
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

st.set_page_config(
    layout="wide"
)

st.sidebar.success("Choisit ton jeux")

#model
model = tf.keras.models.load_model('modail.h5')
test = pd.read_csv("data/test.csv", delimiter=",", dtype='float32')

test_samp = np.array(test.sample(n=1, random_state=np.random.randint(28000)))/255
test_shaping = test_samp.reshape(test_samp.shape[0],*(28,28,1))

col1, col2 = st.columns(2, gap="large")

with col1:

    if st.button("Générer un chiffre"):

        #affichage du chiffre brut
        fig, ax = plt.subplots()
        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        ax.imshow(test_shaping.reshape(28, 28),cmap='gray_r')
        st.pyplot(fig=fig)

        with col2:

            #prédiction
            pred = model.predict(test_shaping)
            pred = (np.round(pred,3)*100).astype(int)
            #résultat
            chiffre_pred = pd.DataFrame(pred).T.idxmax()
            st.write(f'Le chiffre est prédit est {chiffre_pred[0]}')
            #table
            st.table(data=pred)

            st.write("La prédiction est juste ?")

            col11, col12 = st.columns(2,gap="large")

            with col11:
                if st.button("Oui"):
                    st.write('ok cool')


            with col12:
                if st.button("Non"):
                    st.write('Euuh non')
