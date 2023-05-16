#import dependencies

from apikey import apikey 

from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.indexes import VectorstoreIndexCreator
import streamlit as st
from streamlit_chat import message

os.environ['OPENAI_API_KEY'] = apikey
model_id = "gpt-3.5-turbo"

#temporarily load a PDF from some archive
loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")

# Display the page title and the text box for the user to ask the question
st.title('🦜 Search and query academic medical papers ')
prompt = st.text_input("What medical topic would you like to know about?")

