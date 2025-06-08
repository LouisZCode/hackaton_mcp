from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import glob
import os
import csv
import random
from typing import TypedDict, Optional, List, Dict, Literal
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

try:
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None
import yaml

import sys
import io
import contextlib

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to temporarily suppress stderr output."""
    stderr_backup = sys.stderr
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stderr = stderr_backup


load_dotenv()
memory = MemorySaver()

#TODO Implement from langfuse.callback import CallbackHandler

session_id = "11May_Session_06"
user_id = "Luis_Tester"
trace_name = "11May_WebSearch_test4"
#memory:
thread_id = "thread_11May_number_6"

if LANGFUSE_AVAILABLE:
    langfuse_handler = CallbackHandler(
        session_id=session_id, 
        user_id=user_id, 
        trace_name=trace_name
    )
else:
    langfuse_handler = None



import os
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, 'german_teacher_sysprompts.yaml')
with open(yaml_path, 'r') as file:
    prompts = yaml.safe_load(file)

conversation_system_prompt = prompts.get('conversation_mode', '')
analysis_system_prompt = prompts.get('analysis_mode', '')
ReAct_prompt = prompts.get('ReAct_Prompt', '')

# Define session information
session_id = "german_tutor_session"
thread_id = "german_tutor_thread"

web_search_tool = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 2
}

# Initialize the LLM
if LANGFUSE_AVAILABLE and langfuse_handler:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        temperature=0,
            model_kwargs={
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2
            }]
        },
        callbacks=[langfuse_handler]
    )
else:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        temperature=0,
            model_kwargs={
            "tools": [{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 2
            }]
        }
    )

# Define our state structure
class TutorState(TypedDict):
    messages: List[AnyMessage]
    student_name: Optional[str]
    student_id: Optional[str]
    student_level: Optional[str]
    student_hobbies: Optional[List[str]]
    selected_hobby: Optional[str]
    conversation_count: int
    collected_texts: List[str]
    mode: Literal["conversation", "analysis"]

def extract_clean_response(ai_message):
    """Extract only the text parts from a Claude response and fix spacing"""
    if isinstance(ai_message.content, str):
        return ai_message.content
        
    if isinstance(ai_message.content, list):
        # Collect only text content, filter out tool calls and search results
        text_parts = []
        for part in ai_message.content:
            if isinstance(part, dict) and part.get('type') == 'text' and part.get('text'):
                # Add each text part to our collection
                text_parts.append(part.get('text').strip())  # Strip whitespace
        
        # Join text parts and fix spacing issues
        combined_text = " ".join(text_parts)
        # Fix double spaces
        while "  " in combined_text:
            combined_text = combined_text.replace("  ", " ")
        # Fix spacing around punctuation
        for punct in ['.', ',', '!', '?', ':', ';']:
            combined_text = combined_text.replace(f" {punct}", punct)
            combined_text = combined_text.replace(f"{punct} ", f"{punct} ")
        
        return combined_text
    
    return str(ai_message.content)  # Fallback

# Student profile retrieval function
def retrieve_student_profile(student_id, db_path=None):
    """
    Read and return a specific student profile based on ID from the single student_data.csv file.
    """
    # Use script directory if no path provided
    if db_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = script_dir
    
    csv_file = os.path.join(db_path, "student_data.csv")
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"Student data file not found: {csv_file}")
        return False
    
    # Read the CSV file to find the student
    try:
        with open(csv_file, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['id'] == student_id:
                    name = row['name']
                    language_level = row['language_level']
                    hobbies = row['hobbies'].split('|') if row['hobbies'] else []
                    return name, language_level, hobbies
    except Exception as e:
        print(f"Error reading student data file: {e}")
        return False
    
    # If we get here, student ID was not found
    print(f"Student ID {student_id} not found in the system. Please try again.")
    return False

# Function to select a random hobby
def select_random_hobby(hobbies):
    if not hobbies:
        return "general conversation"
    return random.choice(hobbies)

# Node for the tutor's behavior
def tutor_node(state):
    # Get current state
    messages = state.get("messages", [])
    conversation_count = state.get("conversation_count", 0)
    collected_texts = state.get("collected_texts", [])
    mode = state.get("mode", "conversation")
    student_id = state.get("student_id", "unknown")  # Get student ID from state
    
    # Check if there's a new student message to process
    if len(messages) > 0 and isinstance(messages[-1], HumanMessage):
        # Store student message for later analysis
        collected_texts.append(messages[-1].content)
        conversation_count += 1
    
    # Check if we should switch to analysis mode
    if conversation_count >= 3 and mode == "conversation":
        mode = "analysis"
    
    # Prepare the system message based on the mode
    if mode == "conversation":
        # Conversation mode
        hobby = state.get("selected_hobby", "general topics")
        level = state.get("student_level", "A1")
        name = state.get("student_name", "student")
        system_content = f"""You are a German language tutor. The student {name} is at {level} level.
        Have a conversation in German about their hobby: {hobby}.""" + conversation_system_prompt
        
        # Create message list for LLM
        if not messages:
            # First message sends the student ID to the LLM
            llm_messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=f"My ID is {student_id}. My name is {name}. Let's talk about {hobby} in German.")
            ]
        else:
            # Use existing conversation
            llm_messages = [SystemMessage(content=system_content)] + messages
            
            # Make sure we have at least one human message
            if not any(isinstance(msg, HumanMessage) for msg in llm_messages):
                llm_messages.append(HumanMessage(content="Please continue."))
    else:
        # Analysis mode - create a prompt with all collected text
        student_text = "\n\n".join(collected_texts)
        level = state.get("student_level", "A1")
        system_content = f"""You are a German language tutor reviewing a student's text at {level} level.
                        Here is the student's text from our conversation:{student_text}""" + analysis_system_prompt
        
        # Create analysis message
        llm_messages = [
            SystemMessage(content=system_content),
            HumanMessage(content="Please analyze my German text and tell me what errors I made.")
        ]
    
    # Get response from LLM
    with suppress_stderr():
        response = llm.invoke(llm_messages)
    
    # Return updated state
    return {
        "messages": messages + [response],
        "conversation_count": conversation_count,
        "collected_texts": collected_texts,
        "mode": mode
    }

# Build the graph
builder = StateGraph(TutorState)
builder.add_node("tutor", tutor_node)

# Add edges
builder.add_edge(START, "tutor")
builder.add_edge("tutor", END)

# Compile the graph
graph = builder.compile(checkpointer=memory)

# Main function to run the tutor
def main():
    # Ask for student ID
    student_id = input("Please enter your student ID: ")
    
    # Retrieve student profile
    result = retrieve_student_profile(student_id)
    
    if result == False:
        print("Student ID not found. Please try again.")
        return
    
    name, level, hobbies = result
    selected_hobby = select_random_hobby(hobbies)
    
    print(f"Welcome, {name}! Let's practice German by talking about: {selected_hobby}")
    

    initial_state = {
        "messages": [], 
        "student_name": name,
        "student_id": student_id, 
        "student_level": level,
        "student_hobbies": hobbies,
        "selected_hobby": selected_hobby,
        "conversation_count": 0,
        "collected_texts": [],
        "mode": "conversation"
    }
    
    # Start conversation
    current_state = graph.invoke(
        initial_state,
        config={
            "configurable": {"thread_id": thread_id}
        }
    )
    
    ai_message = current_state["messages"][-1]

    clean_response = extract_clean_response(ai_message)
    print(f"\nTutor: {clean_response}")
    
    # Conversation loop
    chatting = True
    while chatting:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["bye", "exit", "quit"]:
            chatting = False
            print("Tutor: Auf Wiedersehen! Goodbye!")
        else:
            # Update the messages with user's new message
            messages = current_state["messages"] + [HumanMessage(content=user_input)]
            
            # Create new state with updated messages
            new_state = {**current_state, "messages": messages}
            
            try:
                current_state = graph.invoke(
                    new_state,
                    config={
                        "configurable": {"thread_id": thread_id},
                        "callbacks": [langfuse_handler]
                    }
                )
            except Exception as e:
                print(f"Langfuse error (continuing): {str(e)[:50]}...")
                # Fall back to invoking without callbacks
                current_state = graph.invoke(
                    new_state,
                    config={"configurable": {"thread_id": thread_id}}
                )

            ai_message = current_state["messages"][-1]
            
            clean_response = extract_clean_response(ai_message)
            print(f"\nTutor: {clean_response}")

if __name__ == "__main__":
    main()