from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parents[1]))

from ollama import chat
from util.llm_utils import pretty_stringify_chat, ollama_seed as seed

# Configure the agent
sign_your_name = 'Pulin Agrawal'
model = 'llama3.2'
options = {
    'temperature': 0.7,
    'max_tokens': 200,
}
messages = [
    {'role': 'system', 'content': 'You are a Dungeon Master for a Dungeons & Dragons game. '
                                  'Engage the player with vivid descriptions, unexpected events, '
                                  'and interactive storytelling. Stay in character and guide the game world.'}
]

# Set a consistent seed for reproducibility
options |= {'seed': seed(sign_your_name)}

# Ensure attempts.txt exists in "Chat History" folder
attempts_folder = Path('ChatHistory')
attempts_folder.mkdir(parents=True, exist_ok=True)
attempts_file = attempts_folder / 'attempts.txt'
attempts_file.touch(exist_ok=True)

# Function to save chat history after each interaction
def save_chat_history():
    with open(attempts_file, 'a') as f:
        file_string  = '\n-------------------------NEW ATTEMPT-------------------------\n\n'
        file_string += f'Model: {model}\n'
        file_string += f'Options: {options}\n'
        file_string += pretty_stringify_chat(messages)
        file_string += '\n------------------------END OF ATTEMPT------------------------\n\n'
        f.write(file_string)
    print("Chat history saved to attempts.txt")

# Chat loop for player interaction
while True:
    user_input = input("You: ")
    if user_input.lower() == '/exit':
        print("Exiting game...")
        save_chat_history()
        break
    
    messages.append({'role': 'user', 'content': user_input})
    response = chat(model=model, messages=messages, stream=False, options=options)
    
    print(f'DM: {response.message.content}')
    messages.append({'role': 'assistant', 'content': response.message.content})
    
    # Save chat history after each response
    save_chat_history()
