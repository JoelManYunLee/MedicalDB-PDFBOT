import os
from apikey import apikey
from langchain.llms import OpenAI
import streamlit as st

os.environ['OPENAI_API_KEY'] = apikey

# Display the page title and the text box for the user to ask the question
st.title('🦜 Search and query academic medical papers ')
prompt = st.text_input("What medical topic would you like to know about?")

llm = OpenAI(temperature = 0.9)

if prompt:
    response = llm(prompt)
    st.write(response)


