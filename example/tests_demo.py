import os
import pytest

from openai import OpenAI

from llmut import LLMUT
from llmut.clients.openai.client_openai_chat import ClientOpenAIChat

from .directory_assistant import DirectoryAssistant


key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=key)
assistant_client = DirectoryAssistant(client)
tester_client = ClientOpenAIChat(client)

LLMUT_ARGS = dict(
    assistant_description='Employee Directory Assistant',
    stop_condition='the assistant provides the user with the required information, any kind of error, or the user gives up in frustration',
    initiate=False,
    tester_client=tester_client,
    assistant_client=assistant_client
)

def test_successful_search_and_fetch():
    LLMUT(
        user_task='Find the room number of Ms. Davis from HR',
        desired_outcome='The user is able to find the room number of Ms. Davis from HR, which is 136',
        **LLMUT_ARGS
    )()

def test_no_such_person():
    LLMUT(
        user_task='Find the room number of Mr. Smith from HR',
        desired_outcome='The user is told that there is no Mr. Smith in the HR department in the directory and no other information is provided',
        **LLMUT_ARGS
    )()

def test_no_such_department():
    LLMUT(
        user_task='Find the room number of Ms. Davis from Marketing',
        desired_outcome='The user is told that there is no such department in the directory and no other information is provided',
        **LLMUT_ARGS
    )()

def test_no_such_person_or_department():
    LLMUT(
        user_task='Find the room number of Mr. Smith from Marketing',
        desired_outcome='The user is told that there is no such person or department in the directory',
        **LLMUT_ARGS
    )()

def test_phone_number():
    LLMUT(
        user_task='Find the phone number of Mr. Brown from IT',
        desired_outcome='The user is able to find the phone number of Mr. Brown from IT, which is 123-456-7898',
        **LLMUT_ARGS
    )()

@pytest.mark.parametrize('execution_number', range(5))
def test_irrelevant_query(execution_number):
    LLMUT(
        user_task='Find the answer to a trivia question (such as: What is the capital of some country? be creative!)',
        desired_outcome='The user is told that the assistant cannot help with that query',
        **LLMUT_ARGS
    )()