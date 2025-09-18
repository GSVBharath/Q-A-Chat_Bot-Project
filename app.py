from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os

from dotenv import load_dotenv
load_dotenv()


from dotenv import load_dotenv
load_dotenv()

# LangSmith Tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A"



# Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant.Please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,engine,temperature,max_tokens):
    llm=Ollama(model=engine,temperature=temperature, num_predict=max_tokens)
    output_parser=StrOutputParser()
    chain = prompt | llm | StrOutputParser()
    answer=chain.invoke({'question':question})
    return answer

# Select the Model
engine=st.sidebar.selectbox("Select the Model",["gemma3:1b","gemma3","mistral"])

#Adjust response parameter
temperature=st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.8)
max_tokens=st.sidebar.slider("Max_Tokens",min_value=50,max_value=300,value=200)

# Main Interface Input
st.write("Go ahead ask any Question you want")
user_input=st.text_input("You:")

if user_input:
    response=generate_response(user_input,engine,temperature,max_tokens)
    st.write(response)
else:st.write("Please provide User Input")    