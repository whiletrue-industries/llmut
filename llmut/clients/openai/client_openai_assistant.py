import json
import atexit 
from openai import OpenAI

from ...client import Client

class ClientOpenAIAssistant(Client):

    def __init__(self, openai_client: OpenAI, 
                 name, instructions, tools, model):
        self.client = openai_client
        self.assistant = openai_client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model
        )
        self.thread = self.client.beta.threads.create()
        self.model = model
        atexit.register(self.cleanup)

    def interact(self, messages):
        if len(messages) > 0:
            role, content = messages[-1]
            assert role == 'user'
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id,
                role=role,
                content=content
            )
        run = self.client.beta.threads.runs.create_and_poll(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,        
        )
        while run.status != 'completed':
            tool_outputs = []
            for tool in run.required_action.submit_tool_outputs.tool_calls:
                if not tool.type == 'function':
                    continue
                arguments = json.loads(tool.function.arguments)
                function_name = tool.function.name
                ret = self.invoke_function(function_name, arguments)
                print(f'>>>> Assistant calling function {function_name}({arguments}) returned {ret}')
                tool_outputs.append(dict(
                    tool_call_id=tool.id,
                    output=json.dumps(ret, ensure_ascii=False)
                ))
            run = self.client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=self.thread.id,
                run_id=run.id,
                tool_outputs=tool_outputs,
            )

        messages = self.client.beta.threads.messages.list(
            thread_id=self.thread.id,
            order='desc'
        )
        for message in messages:
            for block in message.content:
                if block.type == 'text':
                    return block.text.value
        assert False, 'No text block found in assistant response'

    def invoke_function(self, function_name, arguments):
        raise NotImplementedError()

    def cleanup(self):
        try:
            try:
                if self.thread:
                    self.client.beta.threads.delete(self.thread.id)
                    self.thread = None
            except Exception as e:
                print('Error deleting thread', self.thread.id, e)
            try:
                if self.assistant:
                    self.client.beta.assistants.delete(self.assistant.id)
                    self.assistant = None
            except Exception as e:
                print('Error deleting assistant', self.assistant.id, e)
        except KeyboardInterrupt:
            print('Interrupted cleanup, will try again')
            self.cleanup()