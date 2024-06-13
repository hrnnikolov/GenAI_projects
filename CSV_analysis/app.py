
import streamlit as st
from utils import query_agent

st.title('Let\'s do some analysis on your CSV')
st.header('Please upload your CSV file here:')

#capture the CSV file
data = st.file_uploader('Upload CSV file', type='csv')

query = st.text_area('Enter your query')
button = st.button('Generate Response')

if button:
    answer = query_agent(data, query)
    st.write(answer)
