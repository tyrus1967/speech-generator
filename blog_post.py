import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.globals import set_debug

set_debug(True)

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY=""
llm=ChatOpenAI(model="gpt-4o-mini",api_key=OPENAI_API_KEY)
title_prompt= PromptTemplate(
    input_variables=["topic"],
    template ="""
    You are a professional blogger.
    Create an outline for a blog post on the following topic: {topic}
    The outline should include:
        - Introduction
        - 3 main points with subpoints
        - Conclusion
    Generate my topic
    """
    )

speech_prompt= PromptTemplate(
    input_variables=["ABC","language","emotion"],
    template ="""
    You are a professional blogger.
    Write an engaging introduction paragraph based on the following
    outline:{outline}
    The introduction should hook the reader and provide a brief
    overview of the topic. The introduction should use to fully engage the audience
    """
    )
first_chain = title_prompt | llm | StrOutputParser() | (lambda title: (st.write(title),title)[1])

second_chain = speech_prompt | llm

final_chain = first_chain | second_chain

st.title("Speech Generator")
topic = st.text_input("Enter a topic ")
language = st.text_input("Enter a language ")
emotion = st.text_input("Enter an emotion: ")

if topic:
    response = final_chain.invoke({"topic":topic})
    st.write(response.content)
