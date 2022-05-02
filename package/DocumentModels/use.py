
import streamlit as st
from helpers.utilities import try_now

from package.DocumentModels.use_script import infer_use

def code_part():

    with open('./package/DocumentModels/use_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')

def try_out():

    text1, text2, button_clicked = try_now('Universal Sentence Encoder')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_use(text1, text2))