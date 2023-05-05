import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import streamlit_nested_layout
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

#model
model = tf.keras.models.load_model('application/model.h5')

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
    st.write(f'Le modèle est précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}%')


#afficher la fin
if st.session_state.count >= 9:
    st.balloons()
    st.title(f'Bravo tu a un model entrainé, il était précis a {round(sum(st.session_state.stat)*100/len(st.session_state.stat))}%')

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.write("1: Importe ton image sinon yaura l'erreur")
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
    st.write("3: Importe une autre image")


#conversion
image_pred = np.array(ImageOps.invert(image_ported.convert("L")))
resized_img = cv2.resize(image_pred, (28,28)).astype('float32').reshape(1,28,28,1)/255

#affichage resultats en table
st.write('Probabilité du résultat en pourcentage')
y_pred_img = model.predict(resized_img)
y_pred_img = (np.round(y_pred_img,3)*100).astype(int)
pred_table = st.table(data=y_pred_img)

with col3:
    #affichage traitement
    st.write("Ce que voit l'IA")
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
            if st.session_state.count >= 9:
                st.session_state.count = 0
                st.session_state.stat = []
            else:
                st.session_state.count += 1
                st.session_state.stat.append(1)

    with col12:
        #bouton faux
        but0 = st.button("Faux")
        if but0:
            if st.session_state.count >= 9:
                st.session_state.count = 0
                st.session_state.stat = []
            else:
                st.session_state.count += 1
                st.session_state.stat.append(0)

    #affichage du compte de la session et de la func de %
    st.write('Itération = ', st.session_state.count)

    if but1 or but0:
        resultat()

if st.button("Montrer les filtres"):
    successive_outputs = [layer.output for layer in model.layers[1:]]
    visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

    # Let's run input image through our vislauization network
    # to obtain all intermediate representations for the image.
    successive_feature_maps = visualization_model.predict(resized_img.reshape(1, 28, 28, 1))
    # Retrieve are the names of the layers, so can have them as part of our plot
    layer_names = [layer.name for layer in model.layers]
    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        print(feature_map.shape)

        if len(feature_map.shape) == 4:

            # Plot Feature maps for the conv / maxpool layers, not the fully-connected layers

            n_features = feature_map.shape[-1]  # number of features in the feature map
            size       = feature_map.shape[ 1]  # feature map shape (1, size, size, n_features)

            # We will tile our images in this matrix
            display_grid = np.zeros((size, size * n_features))

            # Postprocess the feature to be visually palatable
            for i in range(n_features):
                x  = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std()
                x *=  64
                x += 128
                x  = np.clip(x, 0, 255).astype('uint8')
                # Tile each filter into a horizontal grid
                display_grid[:, i * size : (i + 1) * size] = x
            # Display the grid
                scale = 20. / n_features
            fig = plt.figure( figsize=(scale * n_features, scale) )
            plt.title ( layer_name )
            plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
            plt.grid  ( False )
            plt.imshow( display_grid, aspect='auto', cmap='viridis' )
            st.pyplot(fig)
