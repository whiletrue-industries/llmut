from openai import OpenAI

from ...client import Client

class ClientOpenAIChat(Client):

    def __init__(self, openai_client: OpenAI, model='gpt-3.5-turbo'):
        self.client = openai_client
        self.model = model

    def interact(self, messages):
        messages = [
            dict(role=role, content=content) for role, content in messages
        ]
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return completion.choices[0].message.content