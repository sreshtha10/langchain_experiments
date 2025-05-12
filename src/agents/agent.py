from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.agent_toolkits.load_tools import load_tools
import streamlit as st


llm = ChatOllama(model="llama3.2:3b")

prompt = hub.pull("hwchase17/react")

tools = load_tools(["wikipedia","ddg-search"])

agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)

agent_executer = AgentExecutor(agent=agent,tools=tools,verbose=True)

st.title("AI Agent")
task = st.text_input("Assign me a task!")

if task:
    response = agent_executer.invoke(
        {
            "input":task
        }
    )
    st.write(response["output"])