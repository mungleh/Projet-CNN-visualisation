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
model = tf.keras.models.load_model('application/modail.h5')

#config de la page
st.set_page_config(
    layout="wide"
)

st.sidebar.success("Choisit ton jeux")

#count de la session
if 'count' not in st.session_state:
    st.session_state.count = 0

#stat de la session
if 'stat' not in st.session_state:
    st.session_state.stat = []

#phrase pour afficher le % de précision du model
def resultat():
    st.write(f'Le modèle est précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}%, ptsm')


#quand le compteur atteint 10 itération, le jeux se reset
if st.session_state.count >= 9:
    st.balloons()
    st.write(f'Bravo ta un model entrainé, il était précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}% nulos')
    if st.button("Recommencer le jeux ?"):
        st.session_state.count = 0
        st.session_state.stat = []

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.write("1: Importe ton image starf sinan yaura l'erreur")
    #importeur
    image_port = st.file_uploader("", type=["png", "jpg"])
    #bouton reset
    if st.button("Reset le jeux"):
        st.session_state.count = 0
        st.session_state.stat = []

with col2:
    if image_port:
        image_ported = Image.open(image_port)
        wpercent = (400/float(image_ported.size[0]))
        hsize = 400
        image_ported = image_ported.resize((400,hsize), Image.Resampling.LANCZOS)
        st.image(image_ported)
    st.write("3: Importe une autre image oesh")


#conversion
image_pred = np.array(image_ported.convert('L'))
resized_img = cv2.resize(image_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255

#affichage resultats en table
st.write('Probabilité du résultat en pourcentage')
y_pred_img = model.predict(resized_img)
y_pred_img = (np.round(y_pred_img,3)*100).astype(int)
pred_table = st.table(data=y_pred_img)

with col3:
    #affichage traitement
    st.write("Ce que voit l'IA (ou moi sans lunette)")
    fig, ax = plt.subplots(figsize=(1,1))
    plt.tick_params(left = False, right = False , labelleft = False ,
            labelbottom = False, bottom = False)
    ax.imshow(resized_img[0],cmap='gray')
    st.pyplot(fig=fig)

    #résultat
    chiffre_pred = pd.DataFrame(y_pred_img).T.idxmax()
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

# import pandas as pd
# import numpy as np
# from PIL import Image
# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# import streamlit_nested_layout
# import cv2
# import tensorflow as tf
# import matplotlib.pyplot as plt

# #model
# model = tf.keras.models.load_model('modail.h5')

# st.set_page_config(
#     layout="wide"
# )

# st.sidebar.success("Choisit ton jeux")

# #count de la session
# if 'count' not in st.session_state:
#     st.session_state.count = 0

# #stat de la session
# if 'stat' not in st.session_state:
#     st.session_state.stat = []

# #phrase pour afficher le % de précision du model
# def resultat():
#     st.write(f'Le modèle est précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}%, plus haut que ta beauté')


# #quand le compteur atteint 10 itération, le jeux se reset
# if st.session_state.count >= 9:
#     st.balloons()
#     st.write(f'Bravo ta un model entrainé, il était précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}% nulos')
#     if st.button("Recommencer le jeux ?"):
#         st.session_state.count = 0
#         st.session_state.stat = []

# st.write("Importe ton image sinan sa yaura encore l'érreur")

# col1, col2, col3 = st.columns(3, gap="large")

# with col1:
#     image_port = st.file_uploader("", type=["png", "jpg"])

#     chiffre_pred = []

#     if len(chiffre_pred) == 1:
#         st.write(f'Le chiffre est prédit est {chiffre_pred[0]}')

# with col2:
#     if image_port:
#         image_ported = Image.open(image_port)
#         wpercent = (400/float(image_ported.size[0]))
#         hsize = 400
#         image_ported = image_ported.resize((400,hsize), Image.Resampling.LANCZOS)
#         st.image(image_ported)

# #bouton predict image
# #if st.button('Prédit ton image'):
# #st.balloons()

# #conversion
# image_pred = np.array(image_ported.convert('L'))
# resized_img = cv2.resize(image_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255

# #affichage resultats en table
# st.write('Probabilité du résultat en pourcentage')
# y_pred_img = model.predict(resized_img)
# y_pred_img = (np.round(y_pred_img,3)*100).astype(int)
# pred_table = st.table(data=y_pred_img)
# chiffre_pred = pd.DataFrame(y_pred_img).T.idxmax()

# with col3:
#     #affichage traitement
#     st.write("Ce que voit l'IA")
#     fig1, ax1 = plt.subplots()
#     plt.tick_params(left = False, right = False , labelleft = False ,
#             labelbottom = False, bottom = False)
#     ax1.imshow(resized_img[0],cmap='gray')
#     st.pyplot(fig=fig1)

#     col11, col12 = st.columns(2,gap="large")

#     with col11:
#         #bouton juste
#         but1 = st.button("Juste")
#         if but1:
#             st.session_state.count += 1
#             st.session_state.stat.append(1)

#     with col12:
#         #bouton faux
#         but0 = st.button("Faux")
#         if but0:
#             st.session_state.count += 1
#             st.session_state.stat.append(0)
