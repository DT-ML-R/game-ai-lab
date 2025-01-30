# Prompt Engineering Process

## Step 1

### Intention
The goal of this step was to implement a functional DnD Dungeon Master agent that can interact with the user in a game like style.

### Action/Change
- Used `llama3.2` as the model.
- Added a system message to guide the LLM to act as a Dungeon Master.
- Set `temperature=0.7` for varied responses.
- Increased `max_tokens=200` to allow longer responses.
- Implemented a chat loop to take user input and generate responses.
- Chat history is saved in `ChatHistory\attempts.txt` for review.

### Result
The agent successfully acts as a Dungeon Master, providing descriptions and responding dynamically to player actions. The `/exit` command works as expected to end the game gracefully.

### Reflection/Analysis of the result
The model responded well to the message, staying in character and delivering engaging interactions. The temperature setting most likely made the responses varied, and a little unpredictable. Future iterations might be to adjust temperature or add guardrails for consistency.
