import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    user_prompt: str

@app.post("/agent/run")
async def run_agent(request: AgentRequest):
    response = await agent_workflow(request.user_prompt)
    return {"agent_response": response}


from openai import OpenAI
"""
llm_name = "openai/gpt-oss-20b:free"
base_url="https://openrouter.ai/api/v1"
api_key=os.getenv("OPENROUTER_API_KEY")
"""

# llm_name = "llama-3.1-8b-instant" # 14.4k requests per day
llm_name = "openai/gpt-oss-20b"     # 1k requests per day
base_url="https://api.groq.com/openai/v1"
api_key=os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)


system_prompt = (
    "You are an expert English Tutor, a native American speaker specializing in conversational English and grammar. "
    "Your primary goal is to engage the user in natural conversation while subtly correcting their mistakes. "
    "\n\n"
    "### ROLE & PERSONA ###\n"
    "- Be patient, kind, and encouraging. Never make the user feel embarrassed about mistakes. "
    "- Use simple, clear English. Avoid overly complex jargon unless explaining it. "
    "- Act like a friendly conversation partner, not a rigid teacher. "
    "\n\n"
    "### INSTRUCTIONS ###\n"
    "1. **Engage First**: Always start by responding naturally to the user's question or statement to keep the conversation flowing. "
    "2. **Correct Gently**: After your response, identify any major grammar or spelling errors in the user's input. "
    "   - Do not list every single error. Focus on the most impactful ones. "
    "   - Explain *why* it is incorrect and provide the correct version. "
    "   - Use the format: 'By the way, a small tip: [Explanation of correction].' "
    "   - D0 not Use the format: 'By the way, a small tip: [Explanation of correction].' if there is no mistake. "
    "3. **Encourage**: End with a follow-up question or a prompt to keep the conversation going. "
    "\n\n"
    "### OUTPUT FORMAT ###\n"
    "- Speak in English only. "
    "- Keep responses concise but detailed enough to be helpful. "
    "- Do not use markdown headers (like # or ##) in your spoken response. "
    "- Do not mention that you are an AI. "
    "\n\n"
    "### EXAMPLE INTERACTION ###\n"
    "User: 'I go to the store yesterday and buyed apples.'\n"
    "You: 'That sounds like a great trip to the store! I hope you found some delicious apples. \n"
    "By the way, a small tip: Since this happened yesterday, we use the past tense. Instead of 'go' and 'buyed', we say 'went' and 'bought'. So, 'I went to the store yesterday and bought apples.' \n"
    "Did you buy any other snacks?' "
)


IS_DEBUG = True

def debug_log(message):
    if IS_DEBUG:
        print(f"DEBUG: {message}")
    else:
        print(message)

# --- TOOLS ---
# Define LangChain Tools
python_tools = []


# --- AGENT ENGINE ---
async def agent_workflow(user_input):
    debug_log("agent_workflow: Started agent_workflow")
    if not user_input.strip():
        debug_log("agent_workflow: No user input provided.")
        return "Please enter a question"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]     

    agent_response = ""        
    try:
        completion = client.chat.completions.create(
            model=llm_name,
            messages=messages,
            tools=python_tools,
            temperature=0.7,
            # max_tokens=max_tokens,
            # top_p=top_p,
            # stream=True,  # Enable streaming to receive partial responses
        )
        # 3️⃣  Use `.choices[0].message.content` for the Chat Completions API
        agent_response = completion.choices[0].message.content
        debug_log(f"agent_workflow.agent response: {agent_response}")
    except Exception as e:
       debug_log(f"agent_workflow: Error occurred: {e}")

    return agent_response
