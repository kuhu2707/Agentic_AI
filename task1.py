import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

llm = OllamaLLM(model="llama3")

wiki_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(wiki_client=None))

agent = initialize_agent(
    tools=[wiki_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    max_iterations=5,
    early_stopping_method="generate"
)

st.title(" Wikipedia Agent")
st.write("Ask your question")

user_query = st.text_input("Enter your question:")

if st.button("Ask") and user_query:
    with st.spinner("Thinking..."):
        try:
          
            response = agent.invoke({"input": user_query})
            st.success(response["output"])
        except Exception as e:
            st.error(f"Error: {e}")

