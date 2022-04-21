
import streamlit as st
from helpers.utilities import try_now
from package.PretrainedModels.word2vec_script import infer_word2vec


def code_part():
    st.write('Download the Word2Vec model from the below url and save it to the ./package/PretrainedModels/models.')
    st.write("https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models")
    with open('./package/PretrainedModels/word2vec_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')


def try_out():

    text1, text2, button_clicked = try_now('Word2Vec')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_word2vec(text1, text2))
