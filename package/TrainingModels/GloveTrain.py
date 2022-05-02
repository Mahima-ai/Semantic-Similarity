import streamlit as st


def code_part():
    st.markdown('#### Steps to train the Glove Model')
    st.write("1. Download the latest code available at https://nlp.stanford.edu/projects/glove/")
    st.write("2. Unpack the files:  unzip master.zip")
    st.write("3. Compile the source:  cd GloVe-master && make")
    st.write("4. Open demo.sh, assign the corpus path (dataset to train on) to the CORPUS variable and save the file")
    st.write("5. Run the demo script: ./demo.sh")
    
    st.write("Consult the included README for further usage details")
