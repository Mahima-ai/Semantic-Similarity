
import streamlit as st
from helpers.utilities import try_now
from package.PretrainedModels.FastText_script import infer_fasttext

def code_part():
    st.write('Download the FastText model from the below url and save it to the ./package/PretrainedModels/models.')
    st.write("https://fasttext.cc/docs/en/english-vectors.html")
    with open('./package/PretrainedModels/FastText_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')

def try_out():

    text1,text2,button_clicked = try_now('FastText')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_fasttext(text1, text2))
