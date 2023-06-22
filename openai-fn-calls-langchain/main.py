# import
import os
import json

import openai

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage, ChatMessage, FunctionMessage

from langchain.agents import Tool
from langchain.tools import format_tool_to_openai_function

from dotenv import load_dotenv

# constants
COMPLETION_MODEL = 'gpt-3.5-turbo-0613';
SYSTEM_PROMPT = """
You are a very smart assistant helping users answer their questions.

You goal is to:
- find the best answer for the question in your trained knowledge base
- call the required external function, in case, you don't have an answer
- understand the response from the external function and provide a helpful answer
"""

# define functions
def add_numbers_fn(args):
    # parse input args into two integer numbers
    sargs = arguments.get('__arg1', '')
    aargs = sargs.split(',')
    arg1 = int(aargs[0])
    arg2 = int(aargs[1])
    # logic
    print('add_numbers_fn:', arg1 + arg2)
    return arg1 + arg2

def multiply_numbers_fn(args):
    # parse input args into two integer numbers
    sargs = arguments.get('__arg1', '')
    aargs = sargs.split(',')
    arg1 = int(aargs[0])
    arg2 = int(aargs[1])
    # logic
    print('multiply_numbers_fn:', arg1 * arg2)
    return arg1 * arg2

# get completion function
def get_completion(model, messages, functions):
    return model.predict_messages(messages, functions=functions)

# init
# take environment variables from .env.
load_dotenv()

# variables
# tools
add_numbers_tool = Tool(
    name="add_numbers_fn",
    func=add_numbers_fn,
    description="useful for when you need to add two numbers. Input: two numbers to add. Output: the sum of two numbers",
)

multiply_numbers_tool = Tool(
    name="multiply_numbers_fn",
    func=multiply_numbers_fn,
    description="useful for when you need to multiply two numbers. Input: two numbers to multiply. Output: the multiplication of two numbers",
)
tools = [add_numbers_tool, multiply_numbers_tool]

# describe functions
function_descriptions = [format_tool_to_openai_function(t) for t in tools]
print('function descriptions: ', function_descriptions)

model = ChatOpenAI(model_name="gpt-3.5-turbo-0613", openai_api_key=os.getenv('OPENAI_API_KEY', ''))

# variables
messages = [
    SystemMessage(content=SYSTEM_PROMPT)
]

# logic
while True:
    question = input("Enter a question:")
    messages.append(HumanMessage(content=question))
    print('messages:', messages)
    ai_message = get_completion(model, messages, function_descriptions)
    print('response:', ai_message)

    # case 1: found the ai_message inside model. no need to call external functions
    if ai_message.additional_kwargs.get('function_call', None) is None:
        print('final ai_message:', ai_message.content)
        break    

    # case 2: model decides to make a function call
    if ai_message.additional_kwargs.get('function_call', None):
        fn_name = ai_message.additional_kwargs['function_call'].get('name', '')
        args = ai_message.additional_kwargs['function_call'].get('arguments', '')
        arguments = json.loads(args)
        print(f'calling function {fn_name} with args: {arguments}')
        result = locals()[fn_name](arguments)
        print("function call result:", result)
        # the model must understand the result
        messages.append(ai_message)
        messages.append(FunctionMessage(name=fn_name,content=result))

        ai_message = get_completion(model, messages, function_descriptions) 
        print('function processing:', ai_message)
        print('function processing ai_message:', ai_message.content)

