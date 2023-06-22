# import
import os
import json

import openai

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

# functions
def add_numbers_fn(value1, value2):
    print('add_numbers_fn', value1, value2)
    return value1 + value2

def multiply_numbers_fn(value1, value2):
    print('multiply_numbers_fn', value1, value2)
    return value1 * value2

# get completion function
def get_completion(messages, functions):
    response = openai.ChatCompletion.create(
        model=COMPLETION_MODEL,
        messages=messages,
        functions=functions,
        temperature=0,
    )
    return response

# init
# take environment variables from .env.
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY', '')

# variables
# describe functions
function_descriptions = [
    {
        "name": "add_numbers_fn",
        "description": "function to add two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "value1": {
                    "type": "integer",
                    "description": "The first number to add. For example, 5",
                },
                "value2": {
                    "type": "integer",
                    "description": "The second number to add. For example, 10",
                },
            },
            "required": ["value1", "value2"],
        },
    },
    {
        "name": "multiply_numbers_fn",
        "description": "function to multiply two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "value1": {
                    "type": "integer",
                    "description": "The first number to multiply. For example, 5",
                },
                "value2": {
                    "type": "integer",
                    "description": "The second number to multiply. For example, 10",
                },
            },
            "required": ["value1", "value2"],
        }, 
    }
]

# messages
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
]

# logic
while True:
    question = input("Enter a question:")
    messages.append({"role": "user", "content": question})
    print('messages:', messages)
    response = get_completion(messages, function_descriptions)
    print(response)

    # case 1: found the response inside model. no need to call external functions
      #"choices": [
      #  {
      #    "index": 0,
      #    "message": {
      #      "role": "assistant",
      #      "content": "The capital of France is Paris."
      #    },
      #    "finish_reason": "stop"
      #  }
      #],    
    if response.choices[0]["finish_reason"] == "stop":
        print('final response:', response.choices[0].message["content"])
        break    

    # case 2: model decides to make a function call
      #"choices": [
      #  {
      #    "index": 0,
      #    "message": {
      #      "role": "assistant",
      #      "content": null,
      #      "function_call": {
      #        "name": "add_numbers_fn",
      #        "arguments": "{\n  \"value1\": 2,\n  \"value2\": 1\n}"
      #      }
      #    },
      #    "finish_reason": "function_call"
      #  }
      #],
    if response.choices[0]["finish_reason"] == "function_call":
        fn_name = response.choices[0].message["function_call"].name
        args = response.choices[0].message["function_call"].arguments
        arguments = json.loads(args)        
        result = locals()[fn_name](**arguments)
        print("function call result:", result)
        # the model must understand the result
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": fn_name,
                    "arguments": args,
                },
            }
        )

        messages.append(
            {
                "role": "function", 
                "name": fn_name, 
                "content": f'{{"result": {str(result)} }}'
            }
        )

        response = get_completion(messages, function_descriptions) 
        print('function processing:', response)
        print('function processing response:', response.choices[0].message["content"])

