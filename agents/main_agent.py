"""TBD"""

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

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


load_dotenv()
memory = MemorySaver()

# Load all prompts from YAML file
with open('system_prompts.yaml', 'r') as file:
    prompts = yaml.safe_load(file)

system_prompt = prompts.get('system_prompt', '')
ReAct_prompt = prompts.get('ReAct_Prompt', '')
#print(system_prompt)
#print(ReAct_prompt)

#TODO Make python get the date, and put that date in the session_id, trace_name etc... so it automatically in langfuse tell me the date of the test or

session_id = "june2_test"
user_id = "mcp_hacker"
trace_name = "june2_trace_00"

#memory:
thread_id = "june2_thread_00"
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
def tool_name():
    """
    Tool Description
    """ 
    #return
    pass



tools=[tool_name]
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