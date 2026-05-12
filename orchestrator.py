"""
Health Insurance AI Copilot Orchestrator.
Uses Gemini to reason over user queries and dispatch retrieval tools.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

from config import LLM_MODEL, LLM_TEMPERATURE, SYSTEM_PROMPT
from tools import get_tools

# Load environment variables
load_dotenv()

class Orchestrator:
    def __init__(self):
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment. Please set it in a .env file.")
            
        self.llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            disable_streaming=False
        )
        self.tools = get_tools()
        
        # Create the agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        self.chat_history = []

    def ask(self, query: str) -> str:
        """Process a query and return the synthesized answer."""
        response = self.agent_executor.invoke({
            "input": query,
            "chat_history": self.chat_history
        })
        
        answer = response["output"]
        
        # Update memory (keep last 5 turns)
        self.chat_history.append(("human", query))
        self.chat_history.append(("ai", answer))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
            
        return answer

if __name__ == "__main__":
    # Quick test
    orchestrator = Orchestrator()
    print("\n--- Test Query 1 ---")
    print(orchestrator.ask("What is my deductible for the Gold plan?"))
    print("\n--- Test Query 2 ---")
    print(orchestrator.ask("Does it cover Metformin?"))
