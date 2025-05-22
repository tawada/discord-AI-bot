from anthropic import Anthropic

# Initialize the Anthropic client
anthropic = Anthropic()

# Create a message with Claude-3.7-Sonnet
message = anthropic.messages.create(
    model="claude-3-sonnet-20240229",
    max_tokens=1024,
    temperature=0.7,
    system="You are a helpful AI assistant.",
    messages=[
        {
            "role": "user",
            "content": "What are the main differences between Python and JavaScript?"
        }
    ]
)

# Print the response
print(message.content[0].text)