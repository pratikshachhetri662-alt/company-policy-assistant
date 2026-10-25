import os
import pandas as pd
from groq import Groq
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

# Load credentials from environment variables
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

app = App(token=SLACK_BOT_TOKEN)
groq_client = Groq(api_key=GROQ_API_KEY)

# Load company policies
policies = pd.read_csv("company_policies.csv").fillna("")


@app.event("app_mention")
def handle_mention(event, say):
    question = event.get("text", "")

    # Give the AI the company policies as context
    context = "\n".join(
        f"{row['title']}: {row['policy_text']}"
        for _, row in policies.iterrows()
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a company policy assistant. "
                    "Answer using only the policies below. "
                    "If the answer is not available, say "
                    "'I cannot find an answer in the company policies.' "
                    "Do not invent information.\n\n" + context
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    say(response.choices[0].message.content)


if __name__ == "__main__":
    SocketModeHandler(app, SLACK_APP_TOKEN).start()
