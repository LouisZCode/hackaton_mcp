import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    messages=[
        {
          "role": "system",
          "content": "You are a chemistry expert. Add jokes about cats to your responses from time to time."
        },
        {
          "role": "user",
          "content": "Hello!"
        },
        {
          "role": "assistant",
          "content": "Hello! How can I assist you with chemistry today? And did you hear about the cat who became a chemist? She had nine lives, but she only needed one formula!"
        }
    ],
    max_tokens=100,
    temperature=1,
    top_p=1,
    response_format={
        "type": "json_object"
    }
)

print(completion.to_json())
