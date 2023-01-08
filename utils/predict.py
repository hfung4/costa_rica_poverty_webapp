from cr_analysis.predict import make_prediction
import streamlit as st

# Make prediction on the test data using the persisted trained pipeline
@st.cache()
def get_predictions(input_data):
    test_res = make_prediction(input_data=input_data, is_raw_data=True)
    return test_res
