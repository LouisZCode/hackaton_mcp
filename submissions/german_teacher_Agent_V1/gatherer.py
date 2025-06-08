"""This agent gathers initial information from the user, saves it in a csv and redirects
the user to the first call with a first teacher."""

#for LLM and API Keys
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
#for the save_initial_profile tool:
import string
import os
import datetime
import random
import csv
#For the redirect user tool
import webbrowser
#for the Graph and State:
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph, MessagesState
#giving it memory:
from langgraph.checkpoint.memory import MemorySaver
#For the database / Tools
from langgraph.prebuilt import ToolNode, tools_condition
#for the LangFuse Observability
try:
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    CallbackHandler = None
#For the prompt import:
import yaml


#NOTE   Agent with Memory, Tools and evaluation/Observation into Langgraph

load_dotenv()
memory = MemorySaver()

# Load all prompts from local YAML file
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, 'german_teacher_sysprompts.yaml')
with open(yaml_path, 'r') as file:
    prompts = yaml.safe_load(file)

system_prompt = prompts.get('info_gatherer_system_prompt', '')
ReAct_prompt = prompts.get('ReAct_Prompt', '')
#print(ReAct_prompt)

session_id = "7May_Session_03"
user_id = "Luis_Tester"
trace_name = "7May_SysPrompt_test1"
#memory:
thread_id = "thread_7May_number_3"
if LANGFUSE_AVAILABLE:
    langfuse_handler = CallbackHandler(
        session_id=session_id, 
        user_id=user_id, 
        trace_name=trace_name
    )
else:
    langfuse_handler = None

#NOTE Model:
#best model:   claude-3-7-sonnet-latest
#chep and fast model:  claude-3-5-haiku-latest
if LANGFUSE_AVAILABLE and langfuse_handler:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        temperature=0,
        callbacks=[langfuse_handler]  # Pass callbacks at initialization
    )
else:
    llm = ChatAnthropic(
        model="claude-3-7-sonnet-latest",
        temperature=0
    )

#NOTE TOOLS
def save_initial_profile(name, language_level, hobbies, student_id=None, db_path=None):
    """
    Save a student profile by appending to a single student_data.csv file
    
    Parameters:
    - name: Student's name (string)
    - language_level: Student's language level (string, e.g., "Beginner A1")
    - hobbies: List of up to 3 hobbies (list of strings)
    - student_id: Pre-existing student ID (optional, will generate random alphanumeric if None)
    - db_path: Base directory path where to save the CSV file (default: None - uses script directory)
    
    Returns:
    - student_id: ID of the student
    - filename: Path to the CSV file
    """
    # Use script directory if no path provided
    if db_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = script_dir
    
    # Single CSV file for all students
    csv_file = os.path.join(db_path, "student_data.csv")
    
    # Get current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate student_id if not provided
    if student_id is None:
        chars = string.ascii_uppercase + string.digits
        student_id = ''.join(random.choice(chars) for _ in range(5))
        
        # Ensure the ID is unique by checking existing records
        existing_ids = set()
        if os.path.exists(csv_file):
            with open(csv_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    existing_ids.add(row.get('id', ''))
        
        # Regenerate if there's a collision
        while student_id in existing_ids:
            student_id = ''.join(random.choice(chars) for _ in range(5))
    
    # Format hobbies as a pipe-separated string
    hobby_str = "|".join([h.strip() for h in hobbies if h.strip()])
    
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.exists(csv_file)
    
    # Append student data to CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header only if file is new
        if not file_exists:
            writer.writerow(['id', 'name', 'language_level', 'registration_date', 'hobbies'])
        
        # Write student data
        writer.writerow([student_id, name, language_level, current_time, hobby_str])
    
    return student_id, csv_file

def redirect_user(state):
    """
    Thit tool redirects the user to the webpage stablished in the url
    You dont need to find another website
    """
    url = "https://www.luiszermeno.info/"
    webbrowser.open_new_tab(url)
    return state  

tools=[save_initial_profile, redirect_user]
llm_with_tools = llm.bind_tools(tools)

#Defining Nodes
def gatherer_agent(state: MessagesState):
    return {"messages" : [llm_with_tools.invoke(state["messages"])]}

#'Nodes'
builder = StateGraph(MessagesState)
builder.add_node("gatherer_agent", gatherer_agent)
builder.add_node("tools", ToolNode(tools))

#Edges
builder.add_edge(START, "gatherer_agent")
builder.add_conditional_edges(
    "gatherer_agent",
    tools_condition
)
builder.add_edge("tools", "gatherer_agent")
builder.add_edge("gatherer_agent", END)
graph = builder.compile(checkpointer=memory)


def run_standalone_gatherer():
    """Function to run the gatherer agent in standalone mode"""
    #InitialAnswer
    first_answer = graph.invoke(
        {"messages": [SystemMessage(content=system_prompt),
                      HumanMessage(content="Start the conversation.")]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler]
        }
    )

    # Display AI's first message (greeting/introduction)
    ai_message = first_answer["messages"][-1]
    print(f"\nAI:\n{ai_message.content}")

    #and it continues here:
    chatting = True
    while chatting:

        user_input = input("\nYou:\n")

        
        if user_input == "bye":
            chatting = False

        else:
            final_output = graph.invoke(
        {"messages": first_answer["messages"] + [HumanMessage(content=user_input)]},
        config={
            "configurable": {"thread_id": thread_id},
            "callbacks": [langfuse_handler]
        }
    )
            first_answer = final_output

            # Extract and print just the AI's response
            ai_message = final_output["messages"][-1]  
            print(f"\nAI:\n{ai_message.content}")

# Only run standalone mode if this file is executed directly
if __name__ == "__main__":
    run_standalone_gatherer()