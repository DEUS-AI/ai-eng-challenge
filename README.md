# AI Engineer code challenge

### `1`  Business Requirements – AKA "Why Are We Doing This?" ****

A customer calls the bank, hoping to get help, but instead, they get lost in an endless phone menu maze. Nightmare, right? Well, not on our watch!

In this challenge, you'll build an AI-powered customer support system where multiple LLMs work together like a well-oiled customer service machine. The mission? Identify the customer and send them to the right place without the usual “Please hold while we transfer you” pain.

Here's how our dream team of AI agents rolls:

**Agent 1 – The Greeter**

This LLM is the friendly face (well, metaphorically speaking) of the bank.
It starts the conversation, asks for identification, and makes sure the customer isn’t just a prank caller trying to mess with the system.

**Agent 2 – The Bouncer**

Once Agent 1 verifies the customer, Agent 2 steps in like a VIP club bouncer.
It decides: Regular customer? Premium client? Or... wait, are they even a customer at all?!

**Agent 3 – The Specialist**

If the customer has a specific request (like “Help me with my yacht insurance” 🛥️), this LLM makes sure they get the right expert.

**GuardrailsAI – The Rule Enforcer**

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
```
**Responses examples(yours could be different but take care of user data leaks):**

As you can see the phone is not leaked to non clients, premiun ones has different phone and regular clients doesnt have access to those phones.

```python
# Premium client
# Thank you for reaching out regarding your account issue. 
# As a premium client, we value your experience and are here to assist you.
# For immediate support, please contact our dedicated support department at +1999888999. 
# They will be able to help you promptly and ensure your account is functioning smoothly. Thank you for your patience and understanding.
```

```python
# Regular client
# I'm sorry to hear that you're having trouble with your account. 
# Since you're a regular client, I recommend that you call our support department at +1112112112 for assistance. 
# They will be able to help you resolve the issue promptly. Thank you for your understanding!
```

```python
# Non client
# Thank you for reaching out.
# It seems that you are not currently a client of DEUS Bank. 
# I recommend that you contact your bank's support department directly for assistance with your account issue. 
# They will be able to provide you with the help you need.
```

**Deliverables**
1. A diagram (like the example) to visually show how your system works.
![Graph example](lang-graph.png?raw=true "Graph example")
2. A Pull Request in the repo with your final solution.
3. Realistic commits – because in the real world, we don’t just push all our code in one commit named final_version_for_real_this_time.py.

---

### `3`  Bonus points

Want extra credit (and bragging rights)? Try these:

1. Give your AI a voice – because robotic monotone chat is so last decade.
2. Use fancy guardrails – extra security, extra cool.
3. Make the AI remember past conversations – because forgetting customers is just rude.

Now, go forth and build the most epic AI-powered customer support ever! 🚀
