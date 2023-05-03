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

#config de la page
st.set_page_config(
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
local_css("style.css")

st.sidebar.success("Choisit ton jeux")

#count de la session
if 'count' not in st.session_state:
    st.session_state.count = 0

#stat de la session
if 'stat' not in st.session_state:
    st.session_state.stat = []

#phrase pour afficher le % de précision du model
def resultat():
    st.write(f'Le modèle est précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}% ptsm')


#quand le compteur atteint 10 itération, le jeux se reset
if st.session_state.count >= 9:
    st.balloons()
    st.write(f'Bravo ta un model entrainé, il était précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}% nulos')
    if st.button("Recommencer le jeux ?"):
        st.session_state.count = 0
        st.session_state.stat = []

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.write("Joue jusqua 10 itération pour entrainer ton model oesh")
    #options du canva
    stroke_width = st.slider("Taile du dessinage", 1, 25, 10)
    realtime_update = st.checkbox("Actualisation en live", True)
    #bouton reset
    if st.button("Reset le jeux"):
        st.session_state.count = 0
        st.session_state.stat = []

with col2:
    #canvas
    st.write("1: Déssine puis dit a L'IA ta réponse la")
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

    st.write("3: Clic sur la poubelle pour prédire un nouveau déssin oesh")

#stockage de l'image
if canvas_result.image_data is not None:
    dessin = canvas_result.image_data

#conversion
dessin_pred = np.array(Image.fromarray(dessin).convert('L'))
resized = cv2.resize(dessin_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255

#affichage resultats en table
st.write('Probabilité du résultat en pourcentage')
pred = model.predict(resized)
pred = (np.round(pred,3)*100).astype(int)

#table
st.table(data=pred)


with col3:
    #affichage traitement
    st.write("Ce que voit l'IA (ou moi sans lunette)")
    fig, ax = plt.subplots(figsize=(1,1))
    plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    ax.imshow(resized[0],cmap='gray')
    st.pyplot(fig=fig)

    #résultat
    chiffre_pred = pd.DataFrame(pred).T.idxmax()
    st.write(f'Le chiffre est prédit est {chiffre_pred[0]}')
    st.write("2: Vrai au faux ?")
    col11, col12 = st.columns(2,gap="large")

    with col11:
        #bouton juste
        but1 = st.button("Vrai")
        if but1:
            st.session_state.count += 1
            st.session_state.stat.append(1)

    with col12:
        #bouton faux
        but0 = st.button("Faux")
        if but0:
            st.session_state.count += 1
            st.session_state.stat.append(0)

    #affichage du compte de la session et de la func de %
    st.write('Itération = ', st.session_state.count)

    if but1 or but0:
        resultat()
