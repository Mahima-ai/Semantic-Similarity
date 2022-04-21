
import streamlit as st
from package.PretrainedModels.glove_script import infer_glove
from helpers.utilities import try_now

def code_part():
    st.write('Download the Glove model from the below url and save it to the ./package/PretrainedModels/models.')
    st.write("https://nlp.stanford.edu/projects/glove/")
    with open('package/PretrainedModels/glove_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')


def try_out():

    text1,text2,button_clicked = try_now('Glove')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_glove(text1, text2))
