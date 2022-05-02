import streamlit as st
from helpers.utilities import try_now

from package.OtherModels.crossencoder_script import infer_crossencoder

def code_part():
    st.write('https://www.sbert.net/examples/applications/cross-encoder/README.html')
    with open('./package/OtherModels/crossencoder_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')


def inference():
   
    text1, text2, button_clicked = try_now('Crossencoder')
    st.write('Downloading the model will take more than 2 hours (depending on your network connection)')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_crossencoder(text1, text2))
