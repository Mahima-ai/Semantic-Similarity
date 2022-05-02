
import streamlit as st
import subprocess
import sys
from helpers.utilities import try_now

from package.TrainingModels.FastText_Inference_script import infer_fasttext

def train():
    st.markdown('#### Training Script')
    with open('./package/TrainingModels/FastText_Train_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')

    train_button = st.button(label = 'Train the model')
    
    if train_button:  
        with st.spinner('Training the model...'):      
            matrix = subprocess.run([f"{sys.executable}", "./package/TrainingModels/FastText_Train_script.py"],
                                    capture_output=True,
                                    encoding='utf-8')       
            st.write(matrix.stdout)

def inference():
    st.markdown('#### Inference Script')
    with open('./package/TrainingModels/FastText_Inference_script.py', 'r') as fp:
        data = fp.read()
         
    st.code(data, language='python')

    text1, text2, button_clicked = try_now('Trained FastText')
    if button_clicked:
        with st.spinner('Computing Similarity...'):
            st.markdown(infer_fasttext(text1, text2))
