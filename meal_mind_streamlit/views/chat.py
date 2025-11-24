import streamlit as st
from utils.meal_router_agent import MealRouterAgent
from utils.db import get_user_profile, get_user_inventory, get_latest_meal_plan, get_snowpark_session
from utils.thread_manager import ThreadManager
from utils.feedback_agent import FeedbackAgent
from langchain.schema import HumanMessage, AIMessage
import time

def render_chat(conn, user_id):
    """Render the enhanced chat interface with intelligent routing"""
    
    # Custom CSS for better chat layout
    st.markdown("""
        <style>
        /* Chat container */
        .stChatMessage {
            padding: 1rem !important;
            border-radius: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* User message styling */
        .stChatMessage[data-testid="user-message"] {
            background-color: #262730 !important;
        }
        
        /* Assistant message styling */
        .stChatMessage[data-testid="assistant-message"] {
            background-color: #1e1e1e !important;
        }
        
        /* Chat input */
        .stChatInputContainer {
            border-top: 1px solid #2e2e2e !important;
            padding-top: 1rem !important;
        }
        
        /* Improve readability */
        .stChatMessage p {
            line-height: 1.6 !important;
            margin-bottom: 0.5rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Thread Management
    thread_mgr = ThreadManager(conn)
    
    # Initialize current thread
    if "current_thread_id" not in st.session_state:
        # Create first thread
        thread_id = thread_mgr.create_thread(user_id)
        st.session_state.current_thread_id = thread_id
    
    # Thread Selector UI
    threads = thread_mgr.get_user_threads(user_id, limit=3)
    
    col1, col2, col3 = st.columns([0.6, 0.3, 0.1])
    with col1:
        st.title("💬 Chat with Meal Mind")
    with col2:
        if threads:
            thread_options = {t['thread_id']: t['title'] for t in threads}
            selected = st.selectbox(
                "Conversation",
                options=list(thread_options.keys()),
                format_func=lambda x: thread_options[x][:30] + "...",
                index=0,
                label_visibility="collapsed"
            )
            if selected != st.session_state.current_thread_id:
                st.session_state.current_thread_id = selected
                # Load messages from database for selected thread
                db_messages = thread_mgr.get_thread_messages(selected)
                st.session_state.messages = []
                for msg in db_messages:
                    if msg['role'] == 'user':
                        st.session_state.messages.append(HumanMessage(content=msg['content']))
                    else:
                        st.session_state.messages.append(AIMessage(content=msg['content']))
                # Add welcome message if no messages
                if not st.session_state.messages:
                    st.session_state.messages = [
                        AIMessage(content="Hello! I'm Meal Mind. How can I help you with your nutrition today?")
                    ]
                st.rerun()
    with col3:
        if st.button("➕", help="New conversation"):
            new_thread = thread_mgr.create_thread(user_id)
            st.session_state.current_thread_id = new_thread
            st.session_state.messages = [
                AIMessage(content="Hello! I'm Meal Mind. How can I help you with your nutrition today?")
            ]
            st.rerun()
    
    st.caption("🤖 Ask questions about your meal plan, inventory, or get personalized cooking tips!")
    st.divider()

    # Initialize chat history from database
    if "messages" not in st.session_state or len(st.session_state.messages) == 0:
        # Load messages from current thread
        if st.session_state.current_thread_id:
            db_messages = thread_mgr.get_thread_messages(st.session_state.current_thread_id)
            st.session_state.messages = []
            for msg in db_messages:
                if msg['role'] == 'user':
                    st.session_state.messages.append(HumanMessage(content=msg['content']))
                else:
                    st.session_state.messages.append(AIMessage(content=msg['content']))
        
        # Add welcome message if no messages loaded
        if not st.session_state.messages:
            st.session_state.messages = [
                AIMessage(content="Hello! I'm Meal Mind. How can I help you with your nutrition today?")
            ]

    # Initialize Chat Agent with Router
    if "chat_agent" not in st.session_state:
        session = get_snowpark_session()
        st.session_state.chat_agent = MealRouterAgent(session, conn)

    # Initialize Feedback Agent
    if "feedback_agent" not in st.session_state:
        session = get_snowpark_session()
        st.session_state.feedback_agent = FeedbackAgent(conn, session)

    # Create a container for messages
    message_container = st.container(height=500)
    
    with message_container:
        # Display chat messages
        for i, msg in enumerate(st.session_state.messages):
            if isinstance(msg, HumanMessage):
                with st.chat_message("user", avatar="👤"):
                    st.markdown(msg.content)
            elif isinstance(msg, AIMessage):
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(msg.content)
                    
                    # Add feedback buttons (only for AI messages, not the welcome message)
                    if i > 0:  # Skip welcome message
                        col1, col2, col3 = st.columns([0.1, 0.1, 0.8])
                        with col1:
                            if st.button("👍", key=f"like_{i}", help="I like this response"):
                                st.session_state.feedback_agent.save_explicit_feedback(
                                    user_id=user_id,
                                    entity_id=f"msg_{i}",
                                    entity_name=f"Response about: {st.session_state.messages[i-1].content[:30]}...",
                                    entity_type="ai_response",
                                    feedback="like"
                                )
                                st.success("Thanks for the feedback!")
                        with col2:
                            if st.button("👎", key=f"dislike_{i}", help="I don't like this response"):
                                st.session_state.feedback_agent.save_explicit_feedback(
                                    user_id=user_id,
                                    entity_id=f"msg_{i}",
                                    entity_name=f"Response about: {st.session_state.messages[i-1].content[:30]}...",
                                    entity_type="ai_response",
                                    feedback="dislike"
                                )
                                st.warning("Thanks for the feedback! We'll improve.")

    # Chat Input
    if prompt := st.chat_input("What would you like to know?", key="chat_input"):
        # Add user message to state
        st.session_state.messages.append(HumanMessage(content=prompt))
        
        # Persist user message to database
        thread_mgr.add_message(
            thread_id=st.session_state.current_thread_id,
            role="user",
            content=prompt
        )
        
        # Display user message immediately
        with message_container:
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)

        # Prepare context data
        try:
            # Fetch fresh context
            with st.spinner("Gathering your data..."):
                user_profile = get_user_profile(conn, user_id)
                inventory_df = get_user_inventory(conn, user_id)
                meal_plan_data = get_latest_meal_plan(conn, user_id)
                
                # Format inventory summary
                if not inventory_df.empty:
                    inv_summary = inventory_df.head(20).to_string()  # Limit for token efficiency
                else:
                    inv_summary = "Inventory is empty."
                
                # Format meal plan summary
                if meal_plan_data:
                    mp_summary = str(meal_plan_data.get('meal_plan', {}).get('week_summary', ''))
                    meal_plan_summary = f"Week Summary: {mp_summary}"
                else:
                    meal_plan_summary = "No active meal plan."

                context_data = {
                    "user_profile": user_profile,
                    "inventory_summary": inv_summary,
                    "meal_plan_summary": meal_plan_summary
                }

            # Get streaming response from agent
            with message_container:
                with st.chat_message("assistant", avatar="🤖"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Stream the response
                    for chunk in st.session_state.chat_agent.run_chat_stream(
                        user_input=prompt,
                        user_id=user_id,
                        history=st.session_state.messages[:-1],
                        context_data=context_data
                    ):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.02)  # Slight delay for visual effect
                    
                    # Final response without cursor
                    message_placeholder.markdown(full_response)
            
            # Add assistant response to state
            st.session_state.messages.append(AIMessage(content=full_response))
            
            # Persist assistant message to database
            thread_mgr.add_message(
                thread_id=st.session_state.current_thread_id,
                role="assistant",
                content=full_response
            )
            
            # Generate thread title if this is the first user message
            if len(st.session_state.messages) == 3:  # Welcome + user + assistant
                thread_mgr.generate_thread_title(
                    thread_id=st.session_state.current_thread_id,
                    first_message=prompt,
                    use_llm=True
                )
            
        except Exception as e:
            with message_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.error(f"I encountered an error: {str(e)}")
                    st.info("Please try rephrasing your question or check your connection.")

