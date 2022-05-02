
from asyncio.subprocess import STDOUT
import streamlit as st
import subprocess
import sys

from helpers.utilities import try_now
from package.OtherModels.doc2vec_inference_script import infer_doc2vec


def train():
    st.markdown('#### Training Script')
    with open('./package/OtherModels/doc2vec_train_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')

    train_button = st.button(label='Train the model')

    if train_button:
        with st.spinner('Training the model...'):      
            matrix = subprocess.run([f"{sys.executable}", "./package/OtherModels/doc2vec_train_script.py"],
                                    # stdout = subprocess.PIPE, stderr=STDOUT,
                                    capture_output=True,
                                    encoding='utf-8')
            st.write(matrix.stdout)


def inference():
    st.markdown('#### Inference Script')
    with open('./package/OtherModels/doc2vec_inference_script.py', 'r') as fp:
        data = fp.read()

    st.code(data, language='python')

    text1, text2, button_clicked = try_now('Doc2Vec')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_doc2vec(text1, text2))
