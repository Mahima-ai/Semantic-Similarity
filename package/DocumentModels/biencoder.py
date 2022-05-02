import streamlit as st
from helpers.utilities import try_now
from package.DocumentModels.biencoder_script import infer_biencoder


def code_part():
    st.write('The list of pretrained models available for use is at https://www.sbert.net/docs/pretrained_models.html')
    with open('./package/DocumentModels/biencoder_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')


def inference():

    text1, text2, button_clicked = try_now('Biencoder')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_biencoder(text1, text2))
