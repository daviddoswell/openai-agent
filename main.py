from typing import Sequence, List
from dotenv import load_dotenv
import json
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT

import nest_asyncio
import asyncio 

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
# Load environment variables from a .env file
load_dotenv()

# Define the multiply function
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result integer"""
    return a * b

# Create a FunctionTool for the multiply function
multiply_tool = FunctionTool.from_defaults(fn=multiply)

# Initialize the OpenAI language model
llm = OpenAI(model="gpt-3.5-turbo-0613")
# Create an OpenAIAgent with the defined tools and language model
agent = OpenAIAgent.from_tools(
    [multiply_tool], 
    llm=llm, verbose=True,
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT
)

# Define a custom class for the OpenAIAgent
class MyOpenAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        tools = [
            tool.metadata.to_openai_tool() for _, tool in self._tools.items()
        ]

        ai_message = self._llm.chat(chat_history, tools=tools).message
        additional_kwargs = ai_message.additional_kwargs
        chat_history.append(ai_message)

        tool_calls = ai_message.additional_kwargs.get("tool_calls", None)
        if tool_calls is not None:
            for tool_call in tool_calls:
                function_message = self._call_function(tool_call)
                chat_history.append(function_message)
                ai_message = self._llm.chat(chat_history).message
                chat_history.append(ai_message)

        return ai_message.content

    def _call_function(self, tool_call) -> ChatMessage:
        id_ = tool_call.id
        function_name = tool_call.function.name
        tool_arguments_json = tool_call.function.arguments

        tool_arguments = json.loads(tool_arguments_json)
        tool = self._tools[function_name]

        output = tool(**tool_arguments)

        return ChatMessage(
            name=function_name,
            content=str(output),
            role="tool",
            additional_kwargs={
                "tool_call_id": id_,
                "name": function_name,
            },
        )

# Define the main function for synchronous streaming chat
async def main():
    response = agent.stream_chat(
        "What is 121 * 2? Once you have the answer, use that number to write a story about a group of mice."
    )
    response_gen = response.response_gen

    for token in response_gen:
        print(token, end="")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
