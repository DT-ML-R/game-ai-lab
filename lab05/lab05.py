from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

import random
from util.llm_utils import run_console_chat, tool_tracker

# beauty of Python
@tool_tracker
def process_function_call(function_call):
    name = function_call.name
    args = function_call.arguments
    return globals()[name](**args)

def roll_for(skill, dc, player):
    n_dice = 1
    sides = 20
    roll = sum([random.randint(1, sides) for _ in range(n_dice)])
    if roll >= int(dc):
        return f'{player} rolled {roll} for {skill} and succeeded!'
    else:
        return f'{player} rolled {roll} for {skill} and failed!'

def process_response(self, response):
    """
    Process the LLM response and check for any tool calls.
    If a tool call exists, execute the tool (via process_function_call),
    add the tool's response as a new message, and return the updated response.
    """
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            # Process the tool call and get the result from the corresponding function
            result = process_function_call(tool_call.function)
            # Append the tool's result to the conversation history
            self.messages.append({
                'role': 'tool',
                'name': tool_call.function.name,
                'arguments': tool_call.function.arguments,
                'content': result
            })
        # Generate a new response based on the updated conversation
        return self.completion()
    return response

run_console_chat(template_file=r'C:\Users\labadmin\Documents\spring2025-labs\lab05\lab05_dice_template.json',
                 process_response=process_response)



# Canvas Submission Comment:
# Imagine a virtual travel assistant that not only chats with you but can also interact with multiple APIs:
# It could check flight statuses, book hotels, retrieve weather info, and even calculate travel itineraries in real time.
# With tool calling, such an assistant can delegate specific tasks to the appropriate external service,
# ensuring that the user receives the most current and accurate information seamlessly integrated into the conversation.
