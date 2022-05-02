
from asyncio.subprocess import STDOUT
import streamlit as st
import subprocess
import sys
from helpers.utilities import try_now

from package.OtherModels.bertclassification_inference_script import inference_bert

def train():
    st.write('https://huggingface.co/course/chapter3/3?fw=pt')
    st.markdown('#### Training Script')
    with open('./package/OtherModels/bertclassification_train_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')

    train_button = st.button(label = 'Train the model')
    st.write('Training the model will take more than 2 hours (depending on your machine)')
    if train_button: 
        with st.spinner('Training the model...'):          
            matrix = subprocess.run([f"{sys.executable}", "./package/OtherModels/bertclassification_train_script.py"],
                                    stdout = subprocess.PIPE, stderr=STDOUT,
                                    # capture_output=True,
                                    encoding='utf-8')       
            st.write(matrix.stdout)

def inference():
    st.markdown('#### Inference Script')
    with open('./package/OtherModels/bertclassification_inference_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')

    text1, text2, button_clicked = try_now('Bert as a Classification')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(inference_bert(text1, text2))