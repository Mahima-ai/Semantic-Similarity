import streamlit as st
from helpers.utilities import try_now
from package.PretrainedModels.Bert_script import infer_bert


def code_part():
    st.write('https://huggingface.co/docs/transformers/index')
    with open('./package/PretrainedModels/Bert_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')


def try_out():

    text1, text2, button_clicked = try_now('Bert')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_bert(text1, text2))
