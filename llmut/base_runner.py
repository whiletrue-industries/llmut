from .client import Client
import json


class LLMUT():

    SEP = '--------'

    def __init__(
            self,
            assistant_description,
            user_task,
            stop_condition,
            desired_outcome,
            initiate=False,
            tester_client: Client=None,
            assistant_client: Client=None,
        ):
        self.assistant_description = assistant_description
        self.user_task = user_task
        self.stop_condition = stop_condition
        self.desired_outcome = desired_outcome
        self.initiate = initiate
        self.tester_client = tester_client
        self.assistant_client = assistant_client
        self.tester_messages  = []
        self.assistant_messages = []

    def system_prompt(self):
        ret =f'''
You are to pretend you are a regular user, talking with an AI assistant named "{self.assistant_description}" (I will play the part of the assistant).
Only say the sentences such a user would say, without any other additions or embellishments - and I will provide the assistant responses verbatim.
Don't make up any information that you don't have, and don't look up any information online.
Remember that the assistant is a computer program, so no need to be extra polite or considerate.
When the assistant says goodbye, you should not continue the conversation.'''
        return ret
    
    def first_message(self, message=None):
        ret = self.system_prompt()
        ret += f'''\n
Converse with the assistant until this stop condition is met: {self.stop_condition}.
Once the stop condition is met (or I say goodbye), you _must_ print out "STOP" (and only that) and stop the simulation.
Avoid continuing the conversation if no new information is being exchanged.

Remember - you must behave and sound like a normal user - not an assistant of any kind.
In this specific conversation, the user's (i.e. YOUR) task is: {self.user_task}.'''
        if message:
            ret += f'\n{self.SEP}\n{message}'
        else:
            ret += f'\nPlease start the simulation now:\n{self.SEP}\n'
        return ret
    
    def introspection(self):
        conversation = []
        for role, message in self.assistant_messages:
            if self.SEP in message:
                message = message.split(self.SEP)[-1]
            message = message.strip()
            conversation.append(f'{role}: {message}')
        conversation = '\n'.join(conversation)

        return f'''
Please review the following conversation of a user with an AI assistant and evaluate the performance of the simulated assistant:

{conversation}

- Be critical and honest in your assessment.
- The desired outcome of that conversation was "{self.desired_outcome}".
  Was this desired outcome achieved exactly and in full? 
- Did the assistant make any mistakes or misunderstand the user?
- Was the assistant helpful and efficient?

Please answer as a JSON object with this exact following structure:
{{
    "outcome_accomplished": <true/false, whether desired outcome was achieved in full>,
    "score": <a score from 0-100, indicating how well the user's task was accomplished, e.g. quick and accurate response, no mistakes made etc.>,
    "reason": "<detailed reason why the outcome was accomplished or not, include quotes from the conversation to emphasize your reasoning>"
}}
        ''';

    def __call__(self):
        print('\n-----------------------------------')
        print(f'User Task: {self.user_task}')
        print(f'Desired Outcome: {self.desired_outcome}')
        print('-----------------------------------')
        if self.initiate:
            self.assistant_messages = []
            message = self.run_assistant()
            self.assistant_messages.append(('assistant', message))
            prompt = self.first_message(message)
        else:
            prompt = self.first_message()
        self.tester_messages.append(['user', prompt])
        while True:
            message = self.run_tester()
            self.tester_messages.append(['assistant', message])
            if message == 'STOP' or message.endswith('STOP'):
                break
            self.assistant_messages.append(['user', message])
            message = self.run_assistant()
            self.assistant_messages.append(['assistant', message])
            self.tester_messages.append(['user', message])
        self.tester_messages = [['user', self.introspection()]]
        message = self.run_tester()
        ret = self.parse_introspection(message)
        assert ret['outcome_accomplished'] is True, ret['reason']
        return ret

    def parse_introspection(self, message):
        if not message.startswith('{'):
            message = message[message.index('{'):]
        if not message.endswith('}'):
            message = message[:message.rindex('}')+1]
        return json.loads(message)

    def run_tester(self):
        # print('TesterDDD:', self.tester_messages)
        ret = self.tester_client.interact(self.tester_messages)
        print('> Tester:', ret)
        return ret
    
    def run_assistant(self):
        ret = self.assistant_client.interact(self.assistant_messages)
        print('> Assistant:', ret)
        return ret

