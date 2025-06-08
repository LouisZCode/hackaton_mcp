"""
Gradio interface for the German Language Learning Multi-Agent System
Hackathon submission for Agents-MCP-Hackathon 2025
"""

import gradio as gr
import os
import csv
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Import the gatherer agent components
from gatherer import llm_with_tools, tools, MessagesState, system_prompt, memory
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Import the German conversation agent components
from german_conversaton_agent import (
    TutorState, tutor_node, graph as conversation_graph,
    retrieve_student_profile, select_random_hobby, extract_clean_response
)

class GradioConversationAgent:
    def __init__(self):
        self.memory = MemorySaver()
        self.graph = conversation_graph
        self.current_state = None
        
    def start_conversation(self, student_id):
        """Initialize conversation with student data"""
        result = retrieve_student_profile(student_id)
        
        if not result:
            return None, f"Student ID '{student_id}' not found. Please check your ID or complete registration first."
        
        name, level, hobbies = result
        selected_hobby = select_random_hobby(hobbies)
        
        # Initialize state
        self.current_state = {
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
        try:
            thread_id = f"conversation_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.current_state = self.graph.invoke(
                self.current_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            
            ai_message = self.current_state["messages"][-1]
            clean_response = extract_clean_response(ai_message)
            
            welcome_info = f"""
🎓 **Welcome, {name}!**

📊 **Your Profile:**
- Level: {level}
- Selected Topic: {selected_hobby}
- Interests: {', '.join(hobbies)}

**Let's start your German lesson:**
"""
            
            history = [(welcome_info, clean_response)]
            return history, ""
            
        except Exception as e:
            return None, f"Error starting conversation: {str(e)}"
    
    def chat(self, message, history):
        """Handle chat interaction"""
        if not self.current_state:
            return history, "Please load your profile first."
        
        # Add user message to history
        history.append((message, ""))
        
        try:
            # Update state with new user message
            messages = self.current_state["messages"] + [HumanMessage(content=message)]
            new_state = {**self.current_state, "messages": messages}
            
            # Get response from conversation agent
            thread_id = f"conversation_{self.current_state['student_id']}_{datetime.now().strftime('%Y%m%d')}"
            self.current_state = self.graph.invoke(
                new_state,
                config={"configurable": {"thread_id": thread_id}}
            )
            
            ai_message = self.current_state["messages"][-1]
            clean_response = extract_clean_response(ai_message)
            
            # Check if we're in analysis mode
            if self.current_state.get("mode") == "analysis":
                clean_response = "📝 **Grammar Analysis:**\n\n" + clean_response
            
            # Update history with AI response
            history[-1] = (message, clean_response)
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Sorry, there was an error: {str(e)[:100]}..."
            history[-1] = (message, error_msg)
            return history, ""

class GradioGathererAgent:
    def __init__(self):
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.conversation_history = []
        
    def _build_graph(self):
        """Build the LangGraph for the gatherer agent"""
        def gatherer_agent(state: MessagesState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        builder = StateGraph(MessagesState)
        builder.add_node("gatherer_agent", gatherer_agent)
        builder.add_node("tools", ToolNode(tools))
        
        builder.add_edge(START, "gatherer_agent")
        builder.add_conditional_edges("gatherer_agent", tools_condition)
        builder.add_edge("tools", "gatherer_agent")
        builder.add_edge("gatherer_agent", END)
        
        return builder.compile(checkpointer=self.memory)
    
    def start_conversation(self):
        """Initialize the conversation with the system prompt"""
        thread_id = f"gradio_thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response = self.graph.invoke(
            {"messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Start the conversation.")
            ]},
            config={"configurable": {"thread_id": thread_id}}
        )
        
        ai_message = response["messages"][-1].content
        self.conversation_history = [("", ai_message)]
        return self.conversation_history, thread_id
    
    def chat(self, message, history, thread_id):
        """Handle chat interaction"""
        if not thread_id:
            return history, "Please start a new conversation first."
        
        # Add user message to history
        history.append((message, ""))
        
        try:
            # Use the graph's memory by just adding the new user message
            # The graph will maintain its own conversation state
            response = self.graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config={"configurable": {"thread_id": thread_id}}
            )
            
            ai_response = response["messages"][-1].content
            
            # Check if student ID was created (profile saved)
            student_id = self._extract_student_id(ai_response)
            if student_id:
                ai_response += f"\n\n🎓 **Your Student ID is: {student_id}**\n📝 Please copy this ID and use it in the 'German Class' tab to start your lessons!"
            
            # Update history with AI response
            history[-1] = (message, ai_response)
            
            return history, ""
            
        except Exception as e:
            error_msg = f"Sorry, there was an error: {str(e)}"
            history[-1] = (message, error_msg)
            return history, ""
    
    def _extract_student_id(self, response):
        """Extract student ID from the response if profile was actually saved"""
        # Only return student ID if the response contains confirmation that profile was saved
        if "Student ID has been created" in response or "your Student ID" in response:
            try:
                # Check if the student_data.csv file was recently modified (within last 10 seconds)
                import time
                script_dir = os.path.dirname(os.path.abspath(__file__))
                csv_file = os.path.join(script_dir, "student_data.csv")
                
                if os.path.exists(csv_file):
                    current_time = time.time()
                    file_mod_time = os.path.getmtime(csv_file)
                    
                    # If file was modified within last 10 seconds, get the most recent student ID
                    if current_time - file_mod_time < 10:
                        with open(csv_file, 'r') as file:
                            lines = file.readlines()
                            if len(lines) > 1:  # Has header + at least one data row
                                # Get the last line (most recent student)
                                last_line = lines[-1].strip()
                                if last_line:
                                    # Extract student ID (first column)
                                    student_id = last_line.split(',')[0]
                                    return student_id
            except Exception:
                pass
        return None

def load_student_data(student_id):
    """Load student data from the single student_data.csv file"""
    try:
        # Use the same directory as this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, "student_data.csv")
        
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['id'] == student_id:
                        return row
    except Exception as e:
        return None
    return None

def start_german_class(student_id):
    """Initialize German class with student data"""
    if not student_id:
        return "Please enter your Student ID first."
    
    student_data = load_student_data(student_id)
    if not student_data:
        return f"Student ID '{student_id}' not found. Please check your ID or complete registration first."
    
    welcome_msg = f"""
🎓 Welcome to your German class, {student_data['name']}!

📊 **Your Profile:**
- Level: {student_data['language_level']}
- Interests: {student_data['hobbies'].replace('|', ', ')}
- Registration: {student_data['registration_date']}

🚀 **Ready to start learning German?**
Type 'hello' to begin your first lesson!
    """
    
    return welcome_msg

# Initialize agents lazily to avoid execution during import
gatherer_agent = None
conversation_agent = None

def get_gatherer_agent():
    global gatherer_agent
    if gatherer_agent is None:
        gatherer_agent = GradioGathererAgent()
    return gatherer_agent

def get_conversation_agent():
    global conversation_agent
    if conversation_agent is None:
        conversation_agent = GradioConversationAgent()
    return conversation_agent

# Create Gradio interface
with gr.Blocks(title="German Language Learning - AI Powered", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # German Language Learning - AI Powered
    ### Multi-Agent System for Personalized German Education
    *Hackathon submission for Agents-MCP-Hackathon 2025*
    """)
    
    with gr.Tabs():
        # Tab 1: Registration (Gatherer Agent)
        with gr.TabItem("📝 Registration", id="registration"):
            gr.Markdown("""
            ### Welcome to Luis's AI-Powered German Course!
            First, let's gather some information about you to personalize your learning experience.
            """)
            
            with gr.Row():
                with gr.Column():
                    chatbot = gr.Chatbot(
                        height=400,
                        placeholder="Click 'Start Registration' to begin...",
                        label="Registration Assistant"
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Your message",
                            scale=4
                        )
                        send_btn = gr.Button("Send", scale=1, variant="primary")
                    
                    start_btn = gr.Button("Start Registration", variant="secondary")
                    
                    # Hidden state to store thread_id
                    thread_id_state = gr.State()
        
        # Tab 2: German Class (Conversation Agent)
        with gr.TabItem("🎓 German Class", id="german_class"):
            gr.Markdown("""
            ### Your Personalized German Lessons
            Enter your Student ID from registration to start learning!
            """)
            
            with gr.Row():
                student_id_input = gr.Textbox(
                    placeholder="Enter your 5-character Student ID...",
                    label="Student ID",
                    max_lines=1
                )
                load_btn = gr.Button("Start German Class", variant="primary")
            
            conversation_chatbot = gr.Chatbot(
                height=500,
                placeholder="Enter your Student ID above and click 'Start German Class' to begin your personalized lesson...",
                label="German Tutor"
            )
            
            with gr.Row():
                german_msg = gr.Textbox(
                    placeholder="Type your message in German here...",
                    label="Your message",
                    scale=4
                )
                send_german_btn = gr.Button("Send", scale=1, variant="primary")
            
            gr.Markdown("""
            💡 **Tips:**
            - Try to write 100-300 characters per response
            - After 3 exchanges, you'll get grammar feedback
            - Don't worry about mistakes - that's how you learn!
            """)
            
            # Hidden state for conversation
            conversation_state = gr.State()
    
    # Event handlers for Registration tab
    def start_registration():
        agent = get_gatherer_agent()
        history, thread_id = agent.start_conversation()
        return history, thread_id
    
    start_btn.click(
        start_registration,
        outputs=[chatbot, thread_id_state]
    )
    
    def respond(message, history, thread_id):
        agent = get_gatherer_agent()
        history, clear_msg = agent.chat(message, history, thread_id)
        return history, clear_msg
    
    msg.submit(
        respond,
        inputs=[msg, chatbot, thread_id_state],
        outputs=[chatbot, msg]
    )
    
    send_btn.click(
        respond,
        inputs=[msg, chatbot, thread_id_state],
        outputs=[chatbot, msg]
    )
    
    # Event handlers for German Class tab
    def start_conversation_wrapper(student_id):
        agent = get_conversation_agent()
        history, error = agent.start_conversation(student_id)
        if error:
            return [("Error", error)], ""
        return history, ""
    
    def chat_wrapper(message, history):
        if not message.strip():
            return history, ""
        agent = get_conversation_agent()
        history, clear_msg = agent.chat(message, history)
        return history, clear_msg
    
    load_btn.click(
        start_conversation_wrapper,
        inputs=[student_id_input],
        outputs=[conversation_chatbot, german_msg]
    )
    
    german_msg.submit(
        chat_wrapper,
        inputs=[german_msg, conversation_chatbot],
        outputs=[conversation_chatbot, german_msg]
    )
    
    send_german_btn.click(
        chat_wrapper,
        inputs=[german_msg, conversation_chatbot],
        outputs=[conversation_chatbot, german_msg]
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)