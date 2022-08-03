import pickle
import pandas as pd
import numpy as np
import streamlit as st
import base64
# from PIL import Image


st.set_page_config(page_title='Predicting Fraud in Financial Payment Services!', page_icon="👩‍💻", layout="wide")

html_temp = """
<div style="background-color:#3A5874;padding:10px">
<h1 style="color:white;text-align:center;">Any of your credit card transactions is fradulent?</h1>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)
st.write('\n')

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('g4.png')

st.image("6.jpg")


html_temp = """<br>
<div style="background-color:#3A5874;padding:10px">
<h4 style="color:white;text-align:center;">Getting the list of transactions which are more likely to be fradulent!</h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

st.info(
                f"""
                    By this application you can get the filtered list of fradulent transactions! You only need to give the list of transactions with the required features. Then the applications returns you the list of most likely fradulent transactions!
    
                    """
            )


# Collects user input features into dataframe
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
model_xgb = pickle.load(open("fraud_detection_xgb.pkl","rb"))
model_log=pickle.load(open("fraud_detection_log_smote.pkl","rb"))
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.lower()
    X = df.drop("class", axis =1)    
    proba_log = model_log.predict_proba(X)
    proba_xgb = model_xgb.predict_proba(X)
    df["pred_proba_log"] = proba_log[:,1]
    df["pred_proba_xgb"] = proba_xgb[:,1]
    df = df.round({"pred_proba_log": 2, "pred_proba_xgb": 2})
    df = df.sort_values(by='class', ascending=False)
    #st.write("The number of ")
    st.write(df)
else:
    st.info(
            f"""
                👆 Upload a .csv file first. Sample to try: [Example CSV input file]('https://raw.githubusercontent.com/AriFatih/fraud_detection/main/creditcard_sample.csv')

                """
        )   

    st.stop()