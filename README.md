# AI Engineer code challenge

### `1`  Business Requirements – AKA "Why Are We Doing This?" ****

A customer calls the bank, hoping to get help, but instead, they get lost in an endless phone menu maze. Nightmare, right? Well, not on our watch!

In this challenge, you'll build an AI-powered customer support system where multiple LLMs work together like a well-oiled customer service machine. The mission? Identify the customer and send them to the right place without the usual “Please hold while we transfer you” pain.

Here's how our dream team of AI agents rolls:

Agent 1 – The Greeter

This LLM is the friendly face (well, metaphorically speaking) of the bank.
It starts the conversation, asks for identification, and makes sure the customer isn’t just a prank caller trying to mess with the system.

Agent 2 – The Bouncer

Once Agent 1 verifies the customer, Agent 2 steps in like a VIP club bouncer.
It decides: Regular customer? Premium client? Or... wait, are they even a customer at all?!

Agent 3 – The Specialist 

If the customer has a specific request (like “Help me with my yacht insurance” 🛥️), this LLM makes sure they get the right expert.

GuardrailsAI – The Rule Enforcer

Keeps everything safe, professional, and aligned with bank policies—because, you know, we don’t want an AI accidentally approving million-dollar loans to random strangers.

---

### `2`   Technical Requirements and deliverables

Use LangGraph (or a similar framework) inside a Jupyter Notebook to make these LLMs work together like a dream team.
Each LLM can be any model of your choice (but please, remove API keys from your code before you submit—let’s not accidentally leak secrets, shall we?).

The system should check if at least two out of three details match before proceeding.
```python
example_of_user = {
  "name": "Lisa",
  "phone": "+1122334455",
  "IBAN": "DE89370400440532013000"
}

```java
Example Workflow:

Scene: A customer needs help. Our AI squad is on it!

Customer: "I need help with my account."

Agent 1: “Sure! Can you provide your account number and verify your identity?”

Customer: (Provides details)

Agent 1: “Thank you! Your identity has been verified.”

Agent 2: (Checks status) – “Ooooh, fancy! You’re a premium customer. Redirecting you to our VIP team…”

Agent 3: “Oh, I see you need help with wealth management. Let me connect you with an advisor!”

Final step: The customer is smoothly connected to the right department, no rage-quitting required.
```

**Deliverables**
1. A diagram (like the example) to visually show how your system works. //insert image here
2. A Pull Request in the repo with your final solution.
3. Realistic commits – because in the real world, we don’t just push all our code in one commit named final_version_for_real_this_time.py.

---

### `3`  Bonus points

Want extra credit (and bragging rights)? Try these:

Give your AI a voice – because robotic monotone chat is so last decade.
Use fancy guardrails – extra security, extra cool.
Make the AI remember past conversations – because forgetting customers is just rude.

Now, go forth and build the most epic AI-powered customer support ever! 🚀
