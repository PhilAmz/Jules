import chainlit as cl
import openai
import os
import json # Added
from tools import calculate, search_web, execute_python_code # Added execute_python_code

# Initialize OpenAI client
# The client will automatically pick up the OPENAI_API_KEY from the environment
try:
    client = openai.AsyncOpenAI()
    OPENAI_API_KEY_AVAILABLE = True
except openai.OpenAIError:
    OPENAI_API_KEY_AVAILABLE = False
    # You might want to log this or handle it more gracefully
    print("OpenAI API key not found or invalid. LLM features will be disabled.")

# Tool schemas and available tools
tools_schemas = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calculates the result of a mathematical expression. Use this for any mathematical computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (e.g., '2 + 2 * 8').",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web for information on a given query. Use this to find current information, facts, or specific URLs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'latest AI news', 'Chainlit documentation').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Executes a given string of Python code and returns its output (stdout, stderr, and any errors). Use this tool when the user asks to run Python code, or when you need to perform a computation or task that is best done with Python code. Ensure the code is safe and does not perform harmful operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_string": {
                        "type": "string",
                        "description": "The Python code to execute as a single string. The code should be complete and runnable. For example: 'print(1+1)' or 'for i in range(3): print(i)'."
                    }
                },
                "required": ["code_string"]
            }
        }
    }
]

available_tools = {
    "calculate": calculate,
    "search_web": search_web,
    "execute_python_code": execute_python_code, # Added
}

# Suggestion Actions
SUGGESTION_ACTIONS = [
    cl.Action(name="ask_about_calculator", value="What can you calculate for me?", label="Ask about Calculator", description="Click to ask about calculator tool"),
    cl.Action(name="ask_about_search", value="How do you search the web?", label="Ask about Web Search", description="Click to ask about web search tool"),
    cl.Action(name="simple_math_query", value="What is 100 + 5*10?", label="Simple Math Query", description="Click to send a sample math query"),
    cl.Action(name="chainlit_search_query", value="Can you search for Chainlit documentation?", label="Search for Chainlit Docs", description="Click to search for Chainlit docs")
]

@cl.on_chat_start
async def on_chat_start():
    if not OPENAI_API_KEY_AVAILABLE:
        await cl.Message(content="OpenAI API key not configured. LLM functionality is disabled.").send()
        return

    messages = cl.user_session.get("messages")
    if messages:
        # Display loaded messages
        for msg_data in messages:
            # Determine author based on role, or content if it's a system/tool message for UI
            author = "User"
            if msg_data["role"] == "assistant":
                author = "Assistant"
            elif msg_data["role"] == "tool":
                author = "Tool" # Or "System"
            elif msg_data["role"] == "system" and "tool_call_id" not in msg_data : # Display system prompt if not part of tool call
                 author = "System"


            # For tool messages, content might be structured, adapt for display
            content_to_display = ""
            if isinstance(msg_data["content"], str):
                content_to_display = msg_data["content"]
            elif isinstance(msg_data["content"], list): # Handle potential list content from tool_calls
                content_to_display = json.dumps(msg_data["content"])

            # If it's the assistant's message that initiated a tool call, it might be None or have tool_calls
            if msg_data.get("tool_calls"):
                # Display that a tool was called
                tool_call_info = msg_data["tool_calls"][0] # Assuming one for simplicity
                content_to_display = f"Calling tool: `{tool_call_info.function.name}` with arguments: `{tool_call_info.function.arguments}`"
                author = "Assistant (calling tool)"


            # Only send if there's something to display and it's not a hidden system message
            # (The initial system message is not typically displayed, but handled by on_chat_start directly)
            if content_to_display or msg_data["role"] == "user" or (msg_data["role"] == "assistant" and not msg_data.get("tool_calls")): # Display user messages and direct assistant responses
                 if msg_data["role"] != "system" or os.getenv("DISPLAY_SYSTEM_PROMPTS") == "true": # Avoid displaying raw system prompt unless specified
                    await cl.Message(content=content_to_display, author=author).send()
            elif msg_data["role"] == "tool": # Explicitly display tool responses
                 await cl.Message(content=f"Tool {msg_data.get('name')} responded: {msg_data.get('content')}", author="Tool").send()


        await cl.Message(content="Welcome back! Your chat history is restored.").send()
    else:
        # Initialize messages with a system prompt
        initial_messages = [
            {"role": "system", "content": "You are a helpful assistant. You have access to a calculator, a web search tool, and a Python code executor."}
        ]
        cl.user_session.set("messages", initial_messages)
        # It's often better not to send the system prompt itself to the UI
        await cl.Message(content="Hello! I am your helpful assistant with tools. How can I help you today?").send()

    # Send chat suggestions only if LLM is available
    if OPENAI_API_KEY_AVAILABLE:
        await cl.Message(
            content="Here are some things you can try:",
            actions=SUGGESTION_ACTIONS,
            author="System" # Or "Assistant"
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    if not OPENAI_API_KEY_AVAILABLE:
        await cl.Message(content="Cannot process message: OpenAI API key not configured.").send()
        return

    messages = cl.user_session.get("messages", []) # Ensure messages is a list
    # Append user message first, then save session.
    # This ensures user message is saved even if LLM call fails.
    messages.append({"role": "user", "content": message.content})
    cl.user_session.set("messages", messages)

    try:
        # First call to LLM, providing tools, now with streaming
        response_stream = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=tools_schemas,
            tool_choice="auto",
            stream=True # Enable streaming
        )

        collected_chunks = []
        # Reconstruct the full response message from collected chunks
        reconstructed_response_message = {"role": "assistant", "content": "", "tool_calls": []}
        temp_tool_calls_by_index = {} # To reconstruct tool calls correctly by index

        async for chunk in response_stream:
            collected_chunks.append(chunk) # Save all chunks for reconstruction
            if not chunk.choices: continue
            delta = chunk.choices[0].delta
            if delta.content:
                reconstructed_response_message["content"] += delta.content
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    # Initialize tool call if seeing its index for the first time
                    if tc_delta.index not in temp_tool_calls_by_index:
                        temp_tool_calls_by_index[tc_delta.index] = {
                            "id": tc_delta.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""} # Initialize with empty strings
                        }
                        if tc_delta.function and tc_delta.function.name:
                             temp_tool_calls_by_index[tc_delta.index]["function"]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                             temp_tool_calls_by_index[tc_delta.index]["function"]["arguments"] = tc_delta.function.arguments

                    # Update existing tool call entry
                    else:
                        if tc_delta.id: # Should ideally come first if not already set
                            temp_tool_calls_by_index[tc_delta.index]["id"] = tc_delta.id
                        if tc_delta.function: # function might be None in some deltas
                            if tc_delta.function.name:
                                temp_tool_calls_by_index[tc_delta.index]["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                temp_tool_calls_by_index[tc_delta.index]["function"]["arguments"] += tc_delta.function.arguments

        # Finalize tool_calls list from the reconstructed dictionary
        reconstructed_response_message["tool_calls"] = [temp_tool_calls_by_index[i] for i in sorted(temp_tool_calls_by_index.keys())]
        # Filter out tool_calls that might be incomplete (e.g. missing name or id after reconstruction)
        reconstructed_response_message["tool_calls"] = [
            tc for tc in reconstructed_response_message["tool_calls"]
            if tc.get("id") and tc.get("function", {}).get("name") and isinstance(tc.get("function", {}).get("arguments"), str)
        ]

        final_response_content_to_save = ""

        if reconstructed_response_message["tool_calls"]:
            messages.append(reconstructed_response_message) # Save assistant's intent to call tool(s)

            for tool_call_data in reconstructed_response_message["tool_calls"]:
                function_name = tool_call_data["function"]["name"]
                function_args_str = tool_call_data["function"]["arguments"] # Already a string

                try:
                    function_args = json.loads(function_args_str)
                except json.JSONDecodeError as e:
                    error_msg_content = f"Error decoding arguments for tool `{function_name}`. Arguments: `{function_args_str}`. Error: {e}"
                    await cl.Message(content=error_msg_content, author="Tool Manager", parent_id=message.id).send()
                    messages.append({"role": "tool", "tool_call_id": tool_call_data["id"], "name": function_name, "content": f"Error: Invalid arguments json: {function_args_str}. Details: {e}"})
                    continue

                await cl.Message(content=f"Calling tool: `{function_name}` with arguments: `{json.dumps(function_args)}`", author="Tool Manager", parent_id=message.id).send()

                if function_name in available_tools:
                    tool_function = available_tools[function_name]
                    function_response = tool_function(**function_args)
                    await cl.Message(content=f"Tool `{function_name}` responded: `{function_response}`", author="Tool Manager", parent_id=message.id).send()
                    messages.append({"role": "tool", "tool_call_id": tool_call_data["id"], "name": function_name, "content": function_response})
                else:
                    tool_error_msg = f"Error: Tool `{function_name}` not found by the application."
                    await cl.Message(content=tool_error_msg, author="Tool Manager", parent_id=message.id).send()
                    messages.append({"role": "tool", "tool_call_id": tool_call_data["id"], "name": function_name, "content": tool_error_msg})

            # Second LLM call (this is the one we want to stream to the user)
            llm_response_ui = cl.Message(content="", author="Assistant")
            await llm_response_ui.send()

            second_response_stream = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages, # Now includes tool call results
                stream=True
            )

            full_second_response_content = ""
            async for chunk in second_response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    await llm_response_ui.stream_token(token)
                    full_second_response_content += token

            if not full_second_response_content and llm_response_ui.content == "": # Handle case where stream might be empty
                 await llm_response_ui.stream_token("...") # Default if empty
            await llm_response_ui.update()
            final_response_content_to_save = full_second_response_content

        else:
            # No tool call, direct response from the first stream (which was fully buffered)
            llm_response_ui = cl.Message(content="", author="Assistant")
            await llm_response_ui.send()

            buffered_initial_content = reconstructed_response_message["content"]
            if buffered_initial_content:
                # Simulate streaming the buffered content (word by word for this example)
                # In a true token-by-token for first response, this loop would be different.
                # This is a pragmatic approach for the current subtask structure.
                for char_token in list(buffered_initial_content): # char by char streaming
                    await llm_response_ui.stream_token(char_token)
                await llm_response_ui.update()
            else: # If there was no content and no tool call (e.g. empty message from LLM)
                await llm_response_ui.stream_token("...")
                await llm_response_ui.update()

            final_response_content_to_save = buffered_initial_content

        # Save the final assistant response to history
        if final_response_content_to_save:
            messages.append({"role": "assistant", "content": final_response_content_to_save})
        # Always update session at the end of processing
        cl.user_session.set("messages", messages)

    except openai.APIError as e:
        error_message = f"OpenAI API Error: {e}"
        await cl.Message(content=error_message, author="System").send()
        # User message is already saved, so no specific rollback here unless desired.
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        await cl.Message(content=error_message, author="System").send()

# Action Callbacks
@cl.action_callback("ask_about_calculator") # Name matches cl.Action's name
async def on_ask_about_calculator(action: cl.Action):
    user_message = cl.Message(content=action.value, author="User")
    await on_message(user_message)

@cl.action_callback("ask_about_search")
async def on_ask_about_search(action: cl.Action):
    user_message = cl.Message(content=action.value, author="User")
    await on_message(user_message)

@cl.action_callback("simple_math_query")
async def on_simple_math_query(action: cl.Action):
    user_message = cl.Message(content=action.value, author="User")
    await on_message(user_message)

@cl.action_callback("chainlit_search_query")
async def on_chainlit_search_query(action: cl.Action):
    user_message = cl.Message(content=action.value, author="User")
    await on_message(user_message)
