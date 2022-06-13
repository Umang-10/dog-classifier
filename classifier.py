import streamlit as st 
from fastai.vision.all import *
from PIL import Image
import pathlib
import urllib.request

temp = pathlib.PosixPath
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# MODEL_URL = "https://drive.google.com/uc?export=download&id=1cH5nY1T5oykEcLyWtjA8Wv-Xn8vO20BS"
# urllib.request.urlretrieve(MODEL_URL, "model.pkl")

path = Path()
path.ls(file_exts='.pkl')

learn_inf = load_learner(path/'export.pkl', cpu=True)
# out_pl = st.image(load_image(image), width=250)


def load_image(img_file):
    img = PILImage.create(img_file)
    return img

def on_click_classify(image):
    # load_image(image)
    out_pl = st.image(load_image(image), width=250)
    pred, pred_idx, probs = learn_inf.predict(load_image(image))
    st.write('Prediction: ', str(pred)[10:], '; Probability: ', float(probs[pred_idx]))
    

st.title('Dog Classifier')
st.header('Choose/Click your Dog!!')
image = st.file_uploader(label='Choose your dog!', type=['png', 'jpg'], key='img', help='upload an img of dog')

picture = st.camera_input(label='Click your dog!')


# btn_run.on_change(on_click_classify)
btn_run = st.button(label='Classify')
if btn_run:
    if image: on_click_classify(image)
    else: on_click_classify(picture)
    
st.markdown('#### Created by **Umang Kaushik**')
st.markdown('##### **[Github](https://github.com/Umang-10)**')

