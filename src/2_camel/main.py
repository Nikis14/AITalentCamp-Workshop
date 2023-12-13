import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from dotenv import load_dotenv

from agent import CAMELAgent
from file_tools import read_text

load_dotenv()

assistant_inception_prompt = read_text('prompts/assistant_inception.txt')
user_inception_prompt = read_text('prompts/user_inception.txt')
task_specifier_prompt = read_text('prompts/task_specifier.txt')


def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = \
        assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name,
                                               task=task)[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = \
        user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name,
                                          task=task)[0]

    return assistant_sys_msg, user_sys_msg


def specify_task(task, word_limit):
    task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")

    task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
    task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3))
    task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                                 user_role_name=user_role_name,
                                                                 task=task, word_limit=word_limit)[0]
    specified_task_msg = task_specify_agent.step(task_specifier_msg)
    specified_task = specified_task_msg.content
    return specified_task


def camel_pipeline(user_agent, assistant_agent, initial_assistant_msg):
    chat_turn_limit, n = 5, 0
    assistant_msg = initial_assistant_msg

    # reset agents
    assistant_agent.reset()
    user_agent.reset()

    while n < chat_turn_limit:
        n += 1
        user_ai_msg = user_agent.step(assistant_msg)
        user_msg = HumanMessage(content=user_ai_msg.content)
        # print(f"AI User ({user_role_name}):\n\n{user_msg.content}\n\n")
        # print('-----------------')

        assistant_ai_msg = assistant_agent.step(user_msg)
        assistant_msg = HumanMessage(content=assistant_ai_msg.content)
        # print(f"AI Assistant ({assistant_role_name}):\n\n{assistant_msg.content}\n\n")
        # print('=================')

        # Display the conversation in chat format
        st.text(f"AI User ({user_role_name}):")
        st.info(user_msg.content)
        st.text(f"AI Assistant ({assistant_role_name}):")
        st.success(assistant_msg.content)
        if "<CAMEL_TASK_DONE>" in user_msg.content:
            break


def main(task, word_limit):
    # 0. Specify task
    specified_task = specify_task(task, word_limit)
    print(specified_task)

    # 1. Obtain system messages
    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)

    # 2. Initialize agents
    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2))
    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2))

    # 3. Initialize chat
    assistant_msg = HumanMessage(
        content=(f"{user_sys_msg.content}. "
                 "Now start to give me introductions one by one. "
                 "Only reply with Instruction and Input."))

    # 4. Run pipeline
    camel_pipeline(user_agent, assistant_agent, assistant_msg)


if __name__ == '__main__':
    st.header("Conversation")
    st.title("CAMEL-Langchain Agent")
    st.sidebar.header("Input Settings")
    assistant_role_name = st.sidebar.text_input("Assistant Role Name", "Leading Software Developer")
    user_role_name = st.sidebar.text_input("User Role Name", "Entrepreneur")
    task = st.sidebar.text_area("Task", "Plan necessary tasks to launch AI startup that automates processing of text documents")
    word_limit = st.sidebar.number_input("Word Limit for Task Brainstorming", value=100)
    main(task, word_limit)







