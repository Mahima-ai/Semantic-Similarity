import streamlit as st
from helpers.utilities import try_now
from package.DocumentModels.spacy_script import infer_spacy


def code_part():
    st.write('https://spacy.io/models')
    with open('./package/DocumentModels/spacy_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')


def try_out():

    text1,text2,button_clicked = try_now('Spacy')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_spacy(text1, text2))
