import os
# from apikey import openapikey, serpapikey
from apikeylocal import openapikey, serpapikey
from langchain.llms import OpenAI
from pdfloader import loadPDF
from pdfloader import queryPDF

# For utilizing agent search tools
from langchain.agents import load_tools
from langchain.agents import initialize_agent

# For chaining responses
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

# For frontend user interfacing with app
import streamlit as st

def main():
    os.environ['OPENAI_API_KEY'] = openapikey
    os.environ['SERPAPI_API_KEY'] = serpapikey

    link = "https://arxiv.org/pdf/2302.03803.pdf"    

    llm = OpenAI(temperature = 0.0)

    tool_names = ["serpapi", "arxiv"]
    tools = load_tools(tool_names)
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    # Display the page title and the text box for the user to ask the question
    st.title('🦜 Search and query academic medical papers ')
    prompt = st.text_input(" What medical topic would you like to know about? ")

    # search_template = PromptTemplate(
    #     input_variables=['search'],
    #     template="You are a medical database search bot, search for scholarly and academic papers concerning {search} and grab the PDF link if possible"
    # )

    # search_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

    # search_chain = LLMChain(llm=llm, prompt=search_template, verbose=True, output_key='title', memory=search_memory)

    llm.temperature = 0.9

    if prompt:
        search_results = agent.run(prompt)
        st.write(search_results)
        vectorized_doc = loadPDF(link)
        queryPDF(prompt, vectorized_doc)
        



if __name__ == '__main__':
    main()




