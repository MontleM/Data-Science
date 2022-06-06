import pandas as pd
import numpy as np
import datetime
import time
import pickle
import streamlit as st
from PIL import Image
  
# loading in the model to predict on the data
pickle_in_forest = open('Random_Forest_Classifier.pkl', 'rb')
forest_classifier = pickle.load(pickle_in_forest)

pickle_in_tree = open('Decision_Tree_Classifier.pkl', 'rb')
tree_classifier = pickle.load(pickle_in_tree)
  
def welcome():
    return 'welcome all'
  
# defining the function which will make the prediction using 
# the data which the user inputs
def prediction_forest(sepal_length, sepal_width, petal_length, petal_width):  
   
    prediction = forest_classifier.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

def prediction_tree(sepal_length, sepal_width, petal_length, petal_width):  
   
    prediction = tree_classifier.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    print(prediction)
    return prediction

# this is the main function in which we define our webpage 
def main():
    today = st.date_input("Today\'s Date is",datetime.datetime.now())
    page_options = ["Home","Model Prediction","About"]
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    
    if page_selection == 'Home':
        st.title('Iris Flower Classifier ML App')
        #Display both the original image and the pencil sketch
        catIm = Image.open('flower1.jpg')
        st.image(catIm, width=500, caption="Iris Flower")
        # audio_file = open('Mr JazziQ – Woza Ft. Lady Du, Kabza De Small, Boohle[Fakazagods.com].mp3', 'rb')
        # audio_bytes = audio_file.read()

        # st.audio(audio_bytes, format='audio/ogg')
        
    elif page_selection == 'Model Prediction':
        # uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
        # for uploaded_file in uploaded_files:
        #     bytes_data = uploaded_file.read()
        #     # st.write("filename:", uploaded_file.name)
            # st.write(bytes_data)
            
        # giving the webpage a title
        st.title("Iris Flower Prediction")
        model_options = st.radio("Choose Model",("Random Forest Classifier","Decision Tree Classifier"))

    
        # here we define some of the front end elements of the web page like 
        # the font and background color, the padding and the text to be displayed
        html_temp = """
        <div style ="background-color:dodgerblue;padding:5px;border-radius:10px;">
        <h1 style ="color:black;text-align:center;"><i>Iris Flower Classifier ML App</i></h1>
        </div>
        """

      
        # this line allows us to display the front end aspects we have 
        # defined in the above code
        st.markdown(html_temp, unsafe_allow_html = True)
        
        # the following lines create text boxes in which the user can enter 
        # the data required to make the prediction
        sepal_length = st.text_input("Sepal Length")
        sepal_width = st.text_input("Sepal Width")
        petal_length = st.text_input("Petal Length")
        petal_width = st.text_input("Petal Width")
        result =""
        
        # the below line ensures that when the button called 'Predict' is clicked, 
        # the prediction function defined above is called to make the prediction 
        # and store it in the variable result
        if model_options == "Random Forest Classifier":
            if st.button("Predict"):
                result = prediction_forest(sepal_length, sepal_width, petal_length, petal_width)
                if result == 0:
                    st.success('Iris-setosa')
                elif result == 1:
                    st.success('Iris-versicolor')
                elif result == 2:
                    st.success('Iris-virginica')
                st.balloons()

        if model_options == "Decision Tree Classifier":
            if st.button("Predict"):
                st.balloons()
                result = prediction_tree(sepal_length, sepal_width, petal_length, petal_width)
                if result == 0:
                    st.success('Iris-setosa')
                elif result == 1:
                    st.success('Iris-versicolor')
                elif result == 2:
                    st.success('Iris-virginica')
                st.balloons()
                
    elif page_selection == 'About':
        st.write('Streamlit is a powerful framework for building data apps as well as for ML Apps.\
         It makes it easier to productize your machine learning models. \
         In short, Streamlit empowers data scientists and other forms of data workers to \
         add an interactive UI to any part of the data science and Machine Learning life cycle.')

if __name__=='__main__':
        main()