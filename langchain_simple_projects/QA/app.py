
import streamlit as st
from langchain.llms import OpenAI
import os

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def load_answer(question):
    llm = OpenAI()
    answer = llm(question)
    return answer

st.set_page_config(page_title='LangChain Demo', page_icon=':robot:')
st.gheader('LangChain Demo')

def get_text():
    input_text = st.text_input('You: ', key='input')
    return input_text

user_input = get_text()
response = load_answer(user_input)
submit = st.button('Generate')

if submit:
    st.subheader('Answer:')
    st.write(response)
    
    
