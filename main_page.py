
from importlib.resources import Package
import streamlit as st
from package.PretrainedModels import Glove, Word2Vec, FastText, Bert
from package.TrainingModels import GloveTrain, FastText_Train, Word2Vec_train, bert_train
from package.DocumentModels import use,biencoder, Spacy
from package.OtherModels import BertRegression, doc2vec, crossencoder, bertclassification

st.title('Semantic Similarity')

with st.sidebar:

    
    contents = st.radio(label='Contents',
                        options=('Introduction','Pretrained Word Models','Word Model Training', 
                        'Sentence Embedding Models', 'Additional Models', 'Evaluation', 'References'))

if contents=='Introduction':
    st.header('Introduction')
    with open('./package/Introduction.md','r') as fp:
            st.markdown(fp.read())
elif contents == 'Pretrained Word Models':
    st.header('Pretrained Word Models')
    with st.expander('Semantic Similarity using Pretrained Word Models',expanded=True):
        pretrained_models = st.radio(label='Contents',
                            options=('Introduction','Word2Vec','FastText','Glove','Bert'))
    if pretrained_models == 'Introduction':
        st.subheader("Introduction")
        with open('./package/PretrainedModels/pretrained_models.md','r') as fp:
            st.markdown(fp.read())
    elif pretrained_models == 'Word2Vec':
        st.subheader('Word2Vec')
        Word2Vec.code_part()
        Word2Vec.try_out()
    elif pretrained_models == 'FastText':
        st.subheader('FastText')
        FastText.code_part()
        FastText.try_out()
    elif pretrained_models == 'Glove':
        st.subheader('Glove')
        Glove.code_part()
        Glove.try_out()    
    elif pretrained_models == 'Bert':
        st.subheader('Bert')
        Bert.code_part()
        Bert.try_out()
    
elif contents=='Word Model Training':
    st.header('Word Model Training')
    with st.expander('Training Word Models', expanded=True):
        training_models = st.radio(label = 'Contents',
                                    options=['Introduction','Word2Vec','FastText','Glove','Bert'])
    if training_models == 'Introduction':
        st.subheader('Introduction')
        with open('./package/TrainingModels/training_models.md','r') as fp:
            st.markdown(fp.read())
    elif training_models == 'Word2Vec':
        st.subheader('Word2Vec')
        Word2Vec_train.train()
        Word2Vec_train.inference()
    elif training_models == 'FastText':
        st.subheader('FastText')   
        FastText_Train.train()
        FastText_Train.inference()
    elif training_models == 'Glove':
        st.subheader('Glove')
        GloveTrain.code_part()
    elif training_models == 'Bert':
        st.subheader('Bert')
        bert_train.train()
        bert_train.inference()

elif contents == 'Sentence Embedding Models':
    st.header('Sentence Embedding Models')
    with st.expander('Sentence Embedding Models', expanded=True):
        document_models = st.radio(label='Contents',
                            options=('Introduction','Universal Sentence Encoder','Bi-encoder', 'Spacy'))
    if document_models == 'Introduction':
        st.subheader('Introduction')
        with open('./package/DocumentModels/document_models.md','r') as fp:
            st.markdown(fp.read())   
    elif document_models == 'Bi-encoder':
        st.subheader('Bi-encoder')
        biencoder.code_part()
        biencoder.inference()    
    elif document_models == 'Universal Sentence Encoder':
        st.subheader('Universal Sentence Encoder')
        use.code_part()
        use.try_out()
    elif document_models == 'Spacy':
        st.subheader('Spacy')
        Spacy.code_part()
        Spacy.try_out()

elif contents == 'Additional Models':
    st.header('Models')
    with st.expander('Models', expanded=True):
        other_models = st.radio(label='Contents',
                            options=('Introduction','Doc2Vec','Cross Encoder','Bert Classification','Bert Regression'))
    if other_models == 'Introduction':
        st.subheader('Introduction')
        with open('./package/OtherModels/other_models.md','r') as fp:
            st.markdown(fp.read())
    elif other_models == 'Doc2Vec':
        st.subheader('Doc2Vec')
        doc2vec.train()
        doc2vec.inference()
    elif other_models == 'Cross Encoder':
        st.subheader('Cross Encoder')
        crossencoder.code_part()
        crossencoder.inference()
    elif other_models == 'Bert Classification':
        st.subheader('Bert as a Classification')
        bertclassification.train()
        bertclassification.inference()
    elif other_models == 'Bert Regression':
        st.subheader('Bert as a Regression')
        BertRegression.train()
        BertRegression.inference()
elif contents == 'Evaluation':
    st.subheader('Evaluation')
    with open('./package/Evaluation/Evaluation.md','r') as fp:
        st.markdown(fp.read())
elif contents == 'References':
    st.subheader('References')
    with open('./package/References/reference.md','r') as fp:
            st.markdown(fp.read())