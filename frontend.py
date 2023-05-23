import os
import random
import string

# from apikey import openapikey, serpapikey
from apikey import openapikey, serpapikey
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
from streamlit_chat import message

def main():
    os.environ['OPENAI_API_KEY'] = openapikey
    os.environ['SERPAPI_API_KEY'] = serpapikey    

    llm = OpenAI(temperature = 0.0)

    tool_names =  ["arxiv", "serpapi"]
    tools = load_tools(tool_names)
    agent_memory = ConversationBufferMemory(memory_key='chat_history')
    agent = initialize_agent(tools, llm, agent="conversational-react-description", verbose=True, memory=agent_memory)


    # Display the page title and the text box for the user to ask the question
    st.title('🦜 Search and query academic medical papers ')
    prompt = st.text_input(" What medical topic would you like to know about? ")

    # save the chat history 
    if 'answer' not in st.session_state:
        st.session_state['answer'] = []

    if 'question' not in st.session_state:
        st.session_state['question'] = [] 
 
    if prompt:
        response = agent.run(prompt)

        st.session_state.question.insert(0, prompt)
        st.session_state.answer.insert(0, response)   


        # Display the chat history
        for i in range(len( st.session_state.question)):        
            questionKey = ''.join(random.choice(string.ascii_letters) for i in range(10))
            answerKey = ''.join(random.choice(string.ascii_letters) for i in range(10))
            
            message(st.session_state['question'][i], is_user=True, key= questionKey)
            message(st.session_state['answer'][i], is_user=False, key= answerKey)

if __name__ == '__main__':
    main()




