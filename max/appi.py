import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas

sdbtt1 = st.sidebar.title("IMPORTE TON IMAGE OESH")

image_port = st.sidebar.file_uploader("jenaimar", type=["png", "jpg"])

sdbtt2 = st.sidebar.title("OU ALORS DAISSINE CHEPO")
stroke_width = st.sidebar.slider("Taile dou punsso", 1, 25, 10)
realtime_update = st.sidebar.checkbox("actulise in réel time", True)

if image_port == None:
    st.text("ta pa importé d'image t nul")
else:
    image = Image.open(image_port)
    wpercent = (400/float(image.size[0]))
    hsize = int((float(image.size[1])*float(wpercent)))
    image = image.resize((400,hsize), Image.Resampling.LANCZOS)
    st.image(image)

#canvas
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color='black',
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode='freedraw',
    display_toolbar=True,
    key="canvas",
)
