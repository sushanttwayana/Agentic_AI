import streamlit as st
from v5_database_backend import chatbot, retrive_all_threads, load_all_thread_names, save_thread_name, get_thread_name
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ====================================== UTILITY FUNCTIONS =====================================
# These functions handle thread management, naming, and conversation persistence

def generate_thread_id():
    """Generate a unique thread ID using UUID4 for each new chat session.
    Ensures every conversation has a unique identifier for LangGraph backend."""
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    """Start a completely new chat session:
    1. Generate fresh thread ID
    2. Set as current thread
    3. Clear message history (name auto-generated on first message)
    4. Add to chat threads list"""
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    st.session_state['message_history'] = []
    add_thread(thread_id)
    st.rerun()  # Refresh UI immediately

def add_thread(thread_id):
    """Add thread ID to available conversations list with default name.
    Prevents duplicates and ensures every thread has a display name."""
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    # Default name for new threads (will be updated by first message)
    if thread_id not in st.session_state['thread_names']:
        st.session_state['thread_names'][thread_id] = "New Chat"

def load_conversation(thread_id):
    """Fetch complete conversation history for a specific thread from LangGraph backend.
    Returns list of messages or empty list if conversation doesn't exist."""
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

def generate_conversation_name(user_input):
    """Generate meaningful conversation name from first user message (ChatGPT-style).
    - Truncate to 40 chars for clean sidebar display
    - Capitalize first letter for readability
    - Remove extra whitespace"""
    name = user_input.strip()[:40]
    if name:
        return name.capitalize()
    return "New Chat"

# ====================================== SESSION STATE INITIALIZATION ==========================
# Initialize all required session state variables on first page load or refresh
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []  # Current conversation messages

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()  # Current active thread

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrive_all_threads()  # List of all conversation thread IDs


if 'thread_names' not in st.session_state:
    st.session_state['thread_names'] = load_all_thread_names()  # {thread_id: "Meaningful Conversation Name"}

# Initialize renaming_thread to None to avoid KeyError
if 'renaming_thread' not in st.session_state:
    st.session_state['renaming_thread'] = None

# Ensure current thread exists in threads list with proper name
add_thread(st.session_state['thread_id'])

# ====================================== SIDEBAR UI (Chat Management) ==========================
st.sidebar.title('LangGraph Chatbot')

# NEW CHAT BUTTON - Instantly starts fresh conversation
if st.sidebar.button('New Chat', use_container_width=True):
    reset_chat()

st.sidebar.header('My Conversations')

# Display all conversations (newest first) with rename functionality
for thread_id in st.session_state['chat_threads'][::-1]:
    
    # Always get name from session_state (populated from DB)
    thread_name = st.session_state['thread_names'].get(thread_id, "New Chat")
    
    # Create two columns: main button (80%) + rename button (20%)
    col1, col2 = st.sidebar.columns([4, 1])
    
    with col1:
        # Click to switch to this conversation
        if st.sidebar.button(f"{thread_name}", key=f"select_{thread_id}", use_container_width=True):
            st.session_state['thread_id'] = thread_id
            
            # Load conversation history from LangGraph backend
            messages = load_conversation(thread_id)
            
            # Convert LangGraph message objects to Streamlit format
            temp_messages = []
            for msg in messages:
                role = 'user' if isinstance(msg, HumanMessage) else 'assistant'
                temp_messages.append({'role': role, 'content': msg.content})
            
            st.session_state['message_history'] = temp_messages
            st.rerun()
    
    with col2:
        is_renaming = st.session_state['renaming_thread'] == thread_id
        
        if st.sidebar.button("✏️" if not is_renaming else "❌", 
                           key=f"toggle_rename_{thread_id}", 
                           help="Rename" if not is_renaming else "Cancel"):
            # Toggle rename mode for this thread
            st.session_state['renaming_thread'] = thread_id if not is_renaming else None
            st.rerun()
        
        # Show rename input ONLY when in rename mode for this thread
        if is_renaming:
            new_name = st.sidebar.text_input(
                "New name:", 
                value=thread_name, 
                key=f"rename_input_{thread_id}",
                placeholder="Enter new name..."
            )
            
            if st.sidebar.button("💾 Save", key=f"save_{thread_id}", use_container_width=True):
                # ✅ FIXED: Save name and exit rename mode
                st.session_state['thread_names'][thread_id] = new_name.strip() or "New Chat"
                # Save to BOTH session_state AND DATABASE
                save_thread_name(thread_id, st.session_state['thread_names'][thread_id])
                st.session_state['renaming_thread'] = None  # Exit rename mode
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Auto-named from your first message")

# ====================================== MAIN CHAT INTERFACE ==================================
# Display complete conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

# Chat input field
user_input = st.chat_input('Type your message here...')

if user_input:
    # STEP 1: Auto-generate meaningful name for NEW conversations (ChatGPT-style)
    current_name = st.session_state['thread_names'].get(st.session_state['thread_id'], "")
    if current_name == "New Chat":
        meaningful_name = generate_conversation_name(user_input)
        st.session_state['thread_names'][st.session_state['thread_id']] = meaningful_name

    # STEP 2: Add user message to history and display immediately
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # STEP 3: LangGraph configuration for current thread
    CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    # STEP 4: Stream assistant response in real-time
    with st.chat_message("assistant"):
        def ai_only_stream():
            """Generator that streams ONLY assistant message tokens from LangGraph.
            Filters out system messages and other chunks."""
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content  # Only assistant tokens

        # Progressive streaming to UI
        ai_message = st.write_stream(ai_only_stream())

    # STEP 5: Store complete assistant response in session history
    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})

