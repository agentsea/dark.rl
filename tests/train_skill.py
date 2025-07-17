from orign import Skill, Human, processor


@processor(image="python:3.11-slim", platform="ec2")
def on_feedback(message: Message[Feedback]):
    # Load our actor LLM
    llm = QwenVL2_5.load("playwright-actor")

    # Parse the feedback from the message
    feedback = message.content

    if feedback.approved:
        # Send to the LLM to learn
        llm.learn(feedback.messages)

# Create a human that can review for us. 
# When they do the on_feedback function will be called.
human = Human(
    name="playwright-reviewer",
    medium="ui",
    callback=on_feedback
)


skill = Skill(
    description="Learn to use airbnb",
    examples=[
        "I want to book a room in Tokyo from September 1st to September 3rd",
    ],
    max_steps=20,
    mcp_config={
        "mcpServers": {
            "playwright": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
                "env": {"DISPLAY": ":1"},
            }
        }
    }
)

while True:
    task = skill.sample()

    for step in range(task.max_steps):
        print("\n===\nstep: ", step)
