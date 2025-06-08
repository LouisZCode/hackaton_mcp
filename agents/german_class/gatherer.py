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
from langfuse.callback import CallbackHandler
#For the prompt import:
import yaml


#NOTE   Agent with Memory, Tools and evaluation/Observation into Langgraph

load_dotenv()
memory = MemorySaver()

# Load all prompts from YAML file
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(script_dir, '..', 'system_prompts.yaml')
with open(yaml_path, 'r') as file:
    prompts = yaml.safe_load(file)

system_prompt = prompts.get('info_gatherer_system_prompt', '')
ReAct_prompt = prompts.get('ReAct_Prompt', '')
#print(ReAct_prompt)

#TODO Make python get the date, and put that date in the session_id, trace_name etc... so it automatically in langfuse tell me the date of the test or

session_id = "7May_Session_03"
user_id = "Luis_Tester"
trace_name = "7May_SysPrompt_test1"
#memory:
thread_id = "thread_7May_number_3"
langfuse_handler = CallbackHandler(
    session_id=session_id, 
    user_id=user_id, 
    trace_name=trace_name
)

#NOTE Model:
#best model:   claude-3-7-sonnet-latest
#chep and fast model:  claude-3-5-haiku-latest
llm = ChatAnthropic(
    model="claude-3-7-sonnet-latest",
    temperature=0,
    callbacks=[langfuse_handler]  # Pass callbacks at initialization
)

#NOTE TOOLS
def save_initial_profile(name, language_level, hobbies, student_id=None, db_path="./dummy_student_data"):
    """
    Save a student profile to a CSV file with filename format: dateofcreation_name_ID.csv
    
    Parameters:
    - name: Student's name (string)
    - language_level: Student's language level (string, e.g., "Beginner A1")
    - hobbies: List of up to 3 hobbies (list of strings)
    - student_id: Pre-existing student ID (optional, will generate random alphanumeric if None)
    - db_path: Base directory path where to save the CSV files (default: "./student_data")
    
    Returns:
    - student_id: ID of the student
    - filename: Path to the saved CSV file
    """
    # Create directory if needed
    os.makedirs(db_path, exist_ok=True)
    
    # Get current timestamp for both the filename and data
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Generate student_id if not provided
    if student_id is None:
        # Generate a random 8-character alphanumeric ID
        chars = string.ascii_uppercase + string.digits
        student_id = ''.join(random.choice(chars) for _ in range(5))
        
        # Ensure the ID is unique by checking existing files
        existing_ids = set()
        if os.path.exists(db_path):
            for filename in os.listdir(db_path):
                if filename.endswith('.csv'):
                    try:
                        file_id = filename.split('_')[-1].split('.')[0]
                        existing_ids.add(file_id)
                    except (ValueError, IndexError):
                        pass
        
        # Regenerate if there's a collision (unlikely but possible)
        while student_id in existing_ids:
            student_id = ''.join(random.choice(chars) for _ in range(5))
    
    # Create filename using the required format: dateofcreation_name_ID.csv
    # Replace spaces in name with underscores for the filename
    safe_name = name.replace(' ', '_')
    filename = f"{current_date}_{safe_name}_{student_id}.csv"
    full_path = os.path.join(db_path, filename)
    
    # Format hobbies as a pipe-separated string (up to 3)
    hobby_str = "|".join([h.strip() for h in hobbies[:3] if h.strip()])
    
    # Write to CSV file
    with open(full_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['id', 'name', 'language_level', 'registration_date', 'hobbies'])
        
        # Write student data
        writer.writerow([student_id, name, language_level, current_time, hobby_str])
    
    return student_id, full_path

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