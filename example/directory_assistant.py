import yaml
from pathlib import Path

from openai import OpenAI

from llmut.clients.openai.client_openai_assistant import ClientOpenAIAssistant

FILE_DIR = Path(__file__).parent
EMPLOYEE_DIRECTORY = FILE_DIR / 'employee-directory.yaml'

class DirectoryAssistant(ClientOpenAIAssistant):
    def __init__(self, client: OpenAI):
        super().__init__(
            openai_client=client,
            name='Directory Assistant',
            instructions='Help the user find employee information in the employee directory. Be polite but concise.',
            tools=[
                dict(
                    type='function',
                    function=dict(
                        name='search_by_name',
                        description='Search employees by name',
                        parameters=dict(
                            type='object',
                            properties=dict(
                                query=dict(
                                    type='string',
                                    description='The name (first or last) of the employee to look for. Uses a fuzzy search algorithm.'
                                )
                            ),
                            required=['query']
                        )
                    )
                ),
                dict(
                    type='function',
                    function=dict(
                        name='fetch_record',
                        description='Fetch the record of an employee',
                        parameters=dict(
                            type='object',
                            properties=dict(
                                record_id=dict(
                                    type='string',
                                    description='The unique identifier of the record to fetch.'
                                )
                            ),
                            required=['record_id']
                        )
                    )
                )               
            ],
            model='gpt-4o'
        )
        with EMPLOYEE_DIRECTORY.open('r') as f:
            self.directory = yaml.load(f, Loader=yaml.SafeLoader)
    
    def invoke_function(self, function_name, arguments):
        if function_name == 'search_by_name':
            query = arguments['query'].lower()
            results = []
            for record in self.directory:
                if query in record['name'].lower():
                    results.append(dict(
                        id=record['id'],
                        name=record['name'],
                        department=record['department']
                    ))
            return results
        elif function_name == 'fetch_record':
            record_id = arguments['record_id']
            for record in self.directory:
                if record['id'] == record_id:
                    return record