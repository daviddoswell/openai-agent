### 🙋🏽‍♀️ Tutorial: Building a Multipurpose Chatbot with Python and OpenAI

Welcome to this tutorial! By the end, you'll have a chatbot that can perform basic math, generate creative writing, and even assist with writing in the style of Shakespeare. Let's get started!

### 💯 Part 1: Setup and Basic Multiplication

**Step 1: Install Python 3**

1. Visit the [Python website](https://www.python.org/downloads/).
2. Download the latest version of Python for your operating system.
3. Run the installer and follow the instructions. Make sure to check the box that says "Add Python to PATH".

**👨‍💻 Step 2: Install VS Code**

1. Go to the [VS Code website](https://code.visualstudio.com/).
2. Download and install Visual Studio Code for your OS.

**😍 Step 3: Setup the Project**

1. Open VS Code.
2. Create a new folder for your project.
3. Open a terminal in VS Code (use `Ctrl+``  shortcut).

**🖥️ Step 4: Create a Virtual Environment**

1. In the terminal, run:
    
    ```bash
    python -m venv venv
    
    ```
    
2. Activate the virtual environment:
    - On Windows:
        
        ```bash
        .\\venv\\Scripts\\activate
        
        ```
        
    - On macOS/Linux:
        
        ```bash
        source venv/bin/activate
        
        ```
        

**🗣️ Step 5: Install Required Packages**

1. Install the necessary libraries by running:
    
    ```bash
    pip install openai python-dotenv nest-asyncio llama-index
    
    ```
    

**🔑 Step 6: Get Your OpenAI API Key**

1. Sign up at [OpenAI](https://beta.openai.com/signup/).
2. Go to the API section and create a new API key.

**🌳 Step 7: Create a `.env` File**

1. In your project folder, create a file named `.env`.
2. Add your API key to this file:
    
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    
    ```
    

**✍️ Step 8: Create the Python Script**

1. Create a file named `main.py` in your project folder.
2. Copy and paste the following code:

**🎉 Step 9: Code Explanation and Implementation**

Let's break down the code step by step:

**Imports and Setup:**

```python
from typing import Sequence, List
from dotenv import load_dotenv
import json
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT

import nest_asyncio
import asyncio

nest_asyncio.apply()
load_dotenv()

```

- `typing`: For type annotations.
- `dotenv`: To load environment variables from a `.env` file.
- `json`: To handle JSON data.
- `llama_index`: For integrating with OpenAI's GPT model.
- `nest_asyncio`: To apply patches for running asyncio in Jupyter notebooks.

**Define the Multiplication and Addition Functions:**

```python
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: int, b: int) -> int:
    """Adds two integers and returns the result integer"""
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

```

- `multiply`: Multiplies two integers.
- `add`: Adds two integers.
- `FunctionTool.from_defaults`: Creates a tool for each function that can be used by the agent.

**Initialize the OpenAI Agent:**

```python
llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm, verbose=True,
    system_prompt=SHAKESPEARE_WRITING_ASSISTANT
)

```

- `OpenAI`: Initializes the OpenAI language model.
- `OpenAIAgent.from_tools`: Creates an agent that can use the tools defined earlier.

**Define the Custom Agent Class:**

```python
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

```

- `MyOpenAIAgent`: Custom class to manage interactions and tools.
- `chat`: Manages the chat history and calls functions when needed.
- `_call_function`: Executes the functions as defined tools.

**Main Function to Run the Script:**

```python
async def main():
    response = agent.stream_chat("What is 121 * 2?")
    response_gen = response.response_gen

    for token in response_gen:
        print(token, end="")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())

```

- `main`: Asynchronously handles the main logic to prompt the AI and print the response.

**Running the Script:**

1. In the terminal, run:
    
    ```bash
    python main.py
    
    ```
    

You should see output similar to this:

```
Added user message to memory: What is 121 * 2?
=== Calling Function ===
Calling function: multiply with args: {
  "a": 121,
  "b": 2
}
Got output: 242

```

### 🖊️ Part 2: Creative Writing with the Result

Now let's use the result from the multiplication to write a story.

1. Modify the `main.py` script to include a story generation step:

```python
async def main():
    response = agent.stream_chat(
    "What is 121 * 2? Once you have the answer, use that number to write a"
    " story about a group of mice."
)

    response_gen = response.response_gen

    for token in response_gen:
        print(token, end="")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())

```

1. Run the script again:
    
    ```bash
    python main.py
    
    ```
    

You should see the output and then a story generated by the AI.

### 🧙‍♂️ Part 3: Shakespeare Writing Assistant

Let's extend our chatbot to assist with writing in the style of Shakespeare.

1. Modify the `main.py` script to include a Shakespearean writing assistant:

```python
async def main():
    response = agent.stream_chat(
    "Explain the concept of hash tables using a story device."
)

    response_gen = response.response_gen

    for token in response_gen:
        print(token, end="")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())

```

1. Run the script again:
    
    ```bash
    python main.py
    
    ```
    

You should now see the AI explaining the concept of hash tables in a creative, Shakespearean way.

### 🥳 Congratulations!

You've built a multipurpose chatbot using Python and OpenAI. Keep experimenting and adding new features to make it even more powerful!

---

### Complete Code!

### 1. Simple Multiplication

In this implementation, the agent performs simple multiplication and returns the result.

```python
from typing import Sequence, List
from dotenv import load_dotenv
import json
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.agent.openai import OpenAIAgent

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
    llm=llm, verbose=True
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
    response = agent.stream_chat("What is 121 * 2?")
    response_gen = response.response_gen

    for token in response_gen:
        print(token, end="")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())

```

### 2. Simple Multiplication and Writes a Story

In this implementation, the agent performs simple multiplication and writes a story based on the result.

```python
from typing import Sequence, List
from dotenv import load_dotenv
import json
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.agent.openai import OpenAIAgent

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
    llm=llm, verbose=True
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

```

### 3. Simple Multiplication and Uses the Shakespeare Writing Assistant to Write a Simple Story

In this implementation, the agent performs simple multiplication and uses the Shakespeare Writing Assistant to write a story based on the result.

```python
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

        tool_calls = ai_message

.additional_kwargs.get("tool_calls", None)
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

```

Each implementation builds upon the previous one, adding more complexity and capabilities to the agent's responses. 

The final implementation includes using the Shakespeare Writing Assistant to generate more sophisticated responses.