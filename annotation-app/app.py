import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from google.cloud import storage
from pathlib import Path
from dotenv import load_dotenv
import random
load_dotenv()

DEV_SIMULATION_TIMESTAMP = "2025-06-19_12-34-24"
TEST_SIMULATION_TIMESTAMP = "2025-06-20_12-39-38"

# Initialize session state
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'annotator_email' not in st.session_state:
    st.session_state.annotator_email = None
if 'statements' not in st.session_state:
    # Load statements from JSON file
    statements_path = "data/reddit/cmv/statements.json"
    with open(statements_path, 'r') as f:
        st.session_state.statements = json.load(f)
if 'annotated_files' not in st.session_state:
    st.session_state.annotated_files = {}
if 'annotation_set' not in st.session_state:
    st.session_state.annotation_set = None
if 'simulation_timestamp' not in st.session_state:
    st.session_state.simulation_timestamp = None

# Add custom CSS for Reddit-like styling
st.markdown("""
<style>
    .reddit-comment {
        padding: 8px 0;
        margin: 4px 0;
        border-bottom: 1px solid #EDEFF1;
    }
    .reddit-comment-header {
        color: #7c7c7c;
        font-size: 12px;
        margin-bottom: 4px;
    }
    .reddit-comment-content {
        margin-left: 24px;
        margin-bottom: 8px;
    }
    .reddit-vote-buttons {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-left: 24px;
    }
    .reddit-vote-button {
        background: none;
        border: none;
        color: #878A8C;
        cursor: pointer;
        padding: 0 4px;
    }
    .reddit-vote-button:hover {
        color: #FF4500;
    }
    .reddit-score {
        color: #1A1A1B;
        font-weight: 500;
        font-size: 12px;
        margin: 0 4px;
    }
    .reddit-actions {
        color: #878A8C;
        font-size: 12px;
        margin-left: 8px;
    }
    .reddit-actions a {
        color: #878A8C;
        text-decoration: none;
        margin-right: 8px;
    }
    .reddit-actions a:hover {
        text-decoration: underline;
    }
    .reddit-placeholder {
        opacity: 0.7;
    }
    .reddit-title {
        font-size: 18px;
        font-weight: 500;
        color: #1A1A1B;
        margin-bottom: 16px;
        padding: 8px;
        background-color: #F6F7F8;
        border-radius: 4px;
    }
    /* New styles for conversation list */
    .conversation-row {
        display: flex;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #EDEFF1;
    }
    .conversation-status {
        width: 50px;
        text-align: center;
    }
    .conversation-info {
        flex-grow: 1;
        padding: 0 16px;
        max-height: 60px;
        overflow: hidden;
        text-overflow: ellipsis;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
    }
    .conversation-action {
        width: 100px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Google Cloud Storage setup
def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def get_conversation_files():
    """Get all conversation files from the data directory."""
    set_name = st.session_state.get('annotation_set', 'test')
    if set_name not in ['dev', 'test']:
        set_name = 'test'
    data_dir = Path(f"data/reddit/cmv/{set_name}")
    return sorted(list(data_dir.glob("*.csv")))

def load_conversation(file_path):
    """Load and process a conversation file."""
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    return df

def display_comment(comment, is_placeholder=False):
    """Display a single comment in Reddit-like format."""
    # Format the timestamp
    time_ago = (datetime.now() - comment['Timestamp']).total_seconds()
    if time_ago < 60:
        time_str = "just now"
    elif time_ago < 3600:
        minutes = int(time_ago / 60)
        time_str = f"{minutes}m"
    elif time_ago < 86400:
        hours = int(time_ago / 3600)
        time_str = f"{hours}h"
    elif time_ago < 2592000:  # 30 days
        days = int(time_ago / 86400)
        time_str = f"{days}d"
    elif time_ago < 31536000:  # 365 days
        months = int(time_ago / 2592000)
        time_str = f"{months}mo"
    else:
        years = int(time_ago / 31536000)
        time_str = f"{years}y"

    # Format the comment text to add newline before quotes
    comment_text = comment['Text']
    if comment_text.startswith('>'):
        comment_text = '\n' + comment_text
    # Replace any remaining ">" at the start of lines with a newline
    comment_text = comment_text.replace('\n>', '\n\n>')

    # Create the comment header HTML
    header_html = f"""
    <div class="reddit-comment {'reddit-placeholder' if is_placeholder else ''}">
        <div class="reddit-comment-header">
            <span style="color: #1A1A1B; font-weight: 500;">u/{comment['User Name']}</span>
            <span style="margin: 0 4px;">•</span>
            <span>{time_str}</span>
        </div>
    """
    
    # Create the comment footer HTML
    footer_html = """
        <div class="reddit-vote-buttons">
            <button class="reddit-vote-button">↑</button>
            <span class="reddit-score">1</span>
            <button class="reddit-vote-button">↓</button>
            <div class="reddit-actions">
                <a href="#">Reply</a>
                <a href="#">Share</a>
                <a href="#">Report</a>
                <a href="#">Save</a>
            </div>
        </div>
    </div>
    """
    
    # Display the comment header
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Display the comment text in a styled div
    st.markdown(f'<div class="reddit-comment-content">{comment_text}</div>', unsafe_allow_html=True)
    
    # Display the comment footer
   # st.markdown(footer_html, unsafe_allow_html=True)

# Helper to get user directory (sanitize email for filesystem safety)
def get_user_dir(email):
    safe_email = email.replace('@', '_at_').replace('.', '_dot_')
    return Path(__file__).parent / 'annotations' / safe_email

def load_annotated_files():
    """Load the annotated files for the current user."""
    if not st.session_state.annotator_email:
        return {}
    user_dir = get_user_dir(st.session_state.annotator_email)
    progress_file = user_dir / 'annotation_progress.json'
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_annotated_files():
    """Save the annotated files for the current user."""
    if not st.session_state.annotator_email:
        return
    user_dir = get_user_dir(st.session_state.annotator_email)
    os.makedirs(user_dir, exist_ok=True)
    progress_file = user_dir / 'annotation_progress.json'
    with open(progress_file, 'w') as f:
        json.dump(st.session_state.annotated_files, f)

def get_next_unannotated_file(conversation_files):
    """Get the next file that hasn't been annotated by the current user."""
    if st.session_state.annotator_email not in st.session_state.annotated_files:
        st.session_state.annotated_files[st.session_state.annotator_email] = {}
    
    annotated_files = st.session_state.annotated_files[st.session_state.annotator_email]
    for i, file in enumerate(conversation_files):
        if str(file) not in annotated_files:
            return i
    return 0  # If all files are annotated, start from the beginning

def show_login_page():
    """Display the login page with email input and set selection."""
    st.title("Reddit Conversation Annotation Tool")
    st.write("Please enter your email address to begin annotation.")
    st.write("There is no registration to the app. The email address is just an identifier for me to keep track of the annotators and for you to save your annotations and resume later.")
    st.write("Therefore you can enter a fake address but please remember it 😇")
    
    email = st.text_input("Email Address")
    col1, col2 = st.columns(2)
    with col1:
        dev_clicked = st.button(f"Annotate Dev Set (run {DEV_SIMULATION_TIMESTAMP})")
    with col2:
        test_clicked = st.button(f"Annotate Test Set (run {TEST_SIMULATION_TIMESTAMP})", disabled=True)
    if dev_clicked or test_clicked:
        if email and "@" in email and "." in email.split("@")[1]:  # Better email validation
            st.session_state.annotator_email = email
            st.session_state.annotation_set = 'dev' if dev_clicked else 'test'
            st.session_state.simulation_timestamp = DEV_SIMULATION_TIMESTAMP if dev_clicked else TEST_SIMULATION_TIMESTAMP
            # Load annotated files when user logs in
            st.session_state.annotated_files = load_annotated_files()
            # Set flag to force next unannotated file after login
            st.session_state.force_next_unannotated = True
            st.experimental_rerun()
        else:
            st.error("Please enter a valid email address (e.g., user@example.com)")

def show_conversation_list():
    """Display a list of all conversations with their annotation status."""
    st.title("All Conversations")
    
    # Display annotator info and progress
    st.sidebar.write(f"Annotator: {st.session_state.annotator_email}")
    conversation_files = get_conversation_files()
    annotated_count = len(st.session_state.annotated_files.get(st.session_state.annotator_email, {}))
    total_count = len(conversation_files)
    st.sidebar.write(f"Progress: {annotated_count}/{total_count} conversations annotated")
    
    # Add navigation buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    
    if st.sidebar.button("Back to Annotation"):
        st.session_state.view = "annotation"
        st.rerun()
    
    if st.sidebar.button("Change Email"):
        st.session_state.annotator_email = None
        st.rerun()
    
    # Get all conversations
    if not conversation_files:
        st.error("No conversation files found!")
        return
    
    # Create a search box
    search_query = st.text_input("Search conversations", "")
    
    # Get annotated files for current user
    annotated_files = st.session_state.annotated_files.get(st.session_state.annotator_email, {})
    
    # Filter conversations based on search
    filtered_files = conversation_files
    if search_query:
        filtered_files = [
            f for f in conversation_files 
            if search_query.lower() in f.stem.lower() or 
            search_query.lower() in st.session_state.statements.get(f.stem.split('-')[0].split('_')[1], "").lower()
        ]
    
    # Display each conversation
    for i, file in enumerate(filtered_files):
        # Get conversation ID, thread ID and title
        # File format: submission_<conversation_id>-thread_<thread_id>.csv
        file_parts = file.stem.split('-')
        conversation_id = file_parts[0].split('_')[1]
        thread_id = file_parts[1].split('_')[1]
        title = st.session_state.statements.get(conversation_id, "Title not found")
        
        # Remove "CMV:" prefix if it exists
        if title.startswith("CMV:"):
            title = title[4:].strip()
        
        # Check if this conversation has been annotated
        is_annotated = str(file) in annotated_files
        
        # Create columns for each conversation
        col1, col2, col3 = st.columns([1, 4, 1])
        
        # Status
        col1.write("✅" if is_annotated else "⭕")
        
        # Conversation info
        safe_title = title.replace('*', '\\*')
        col2.markdown(f"**{safe_title}**")
        col2.markdown(f"*Submission ID:* {conversation_id}  \n*Thread ID:* {thread_id}")
        
        # View button
        if col3.button("View", key=f"view_{i}"):
            # Find the index of this conversation in the full list
            for j, f in enumerate(conversation_files):
                if f == file:
                    st.session_state.current_file_index = j
                    st.session_state.view = "annotation"
                    st.rerun()
                    break

def show_annotation_interface():
    """Display the main annotation interface."""
    st.title("Reddit Conversation Annotation Tool")
    
    # Display annotator info and progress
    st.sidebar.write(f"Annotator: {st.session_state.annotator_email}")
    conversation_files = get_conversation_files()
    annotated_files_dict = st.session_state.annotated_files.get(st.session_state.annotator_email, {})
    annotated_files = list(annotated_files_dict.keys())
    annotated_count = len(annotated_files)
    total_count = len(conversation_files)
    st.sidebar.write(f"Progress: {annotated_count}/{total_count} conversations annotated")
    
    # Add navigation buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")

    # --- Add Previous and Next Buttons to Sidebar ---
    current_file = conversation_files[st.session_state.current_file_index]
    # Previous button: always go to previous thread, disabled for first thread
    prev_disabled = st.session_state.current_file_index == 0
    next_disabled = st.session_state.current_file_index >= len(conversation_files) - 1
    if st.sidebar.button("Previous", disabled=prev_disabled):
        if st.session_state.current_file_index > 0:
            st.session_state.current_file_index -= 1
            st.experimental_rerun()
    if st.sidebar.button("Next", disabled=next_disabled):
        if st.session_state.current_file_index < len(conversation_files) - 1:
            st.session_state.current_file_index += 1
            st.experimental_rerun()

    if st.sidebar.button("View All Conversations"):
        st.session_state.view = "list"
        st.rerun()
    
    if st.sidebar.button("Change Email"):
        st.session_state.annotator_email = None
        st.rerun()
    
    if not conversation_files:
        st.error("No conversation files found!")
        return
    
    # Load current conversation
    current_file = conversation_files[st.session_state.current_file_index]
    
    # Extract submission ID and thread ID from filename
    file_parts = current_file.stem.split('-')
    conversation_id = file_parts[0].split('_')[1]
    thread_id = file_parts[1].split('_')[1]
    title = st.session_state.statements.get(conversation_id, "Title not found")
    
    # Remove "CMV:" prefix if it exists
    if title.startswith("CMV:"):
        title = title[4:].strip()
    
    # Display the title and IDs
    st.markdown(f'<div class="reddit-title">CMV: {title}</div>', unsafe_allow_html=True)
    thread_number = st.session_state.current_file_index + 1
    is_annotated = str(current_file) in annotated_files
    status_emoji = '✅' if is_annotated else '⭕'
    st.markdown(
        f'**Submission ID:** {conversation_id} | **Thread ID:** {thread_id} | **Thread #{thread_number}/{total_count}** {status_emoji}',
        unsafe_allow_html=True
    )

   
    df = load_conversation(current_file)
    
    # Display conversation history (excluding last 2 comments)
    st.subheader("Conversation History")
    for _, comment in df[:-2].iterrows():
        display_comment(comment)
    
    # Randomize left/right assignment for actual vs generated comments
    if 'comment_side_order' not in st.session_state or st.session_state.get('last_slider_file', None) != str(current_file):
        random.seed(f"{conversation_id}_{thread_id}")
        st.session_state.comment_side_order = random.choice(['actual_left', 'generated_left'])
        st.session_state.last_slider_file = str(current_file)
    side_order = st.session_state.comment_side_order

    actual_comments = list(df[-2:].iterrows())
    # Prepare generated comments (as before)
    generated_comments = []
    simulation_timestamp = st.session_state.simulation_timestamp
    generated_transcript_dir = f"generated_debates/{st.session_state.annotation_set}/nojson_nofs_profiles/{simulation_timestamp}/transcripts"
    generated_filename = f"sub_{conversation_id}-comment_{thread_id}.txt"
    generated_path = os.path.join(generated_transcript_dir, generated_filename)
    print(generated_path)
    if os.path.exists(generated_path):
        with open(generated_path, "r") as f:
            lines = f.readlines()
        import re
        msg_lines = [line.strip() for line in lines if re.match(r"^\d{2}:\d{2}:\d{2} - .+?:", line)]
        last_two = msg_lines[-2:] if len(msg_lines) >= 2 else msg_lines
        actual_timestamps = [comment[1]['Timestamp'] for comment in actual_comments]
        for i, line in enumerate(last_two):
            match = re.match(r"^(\d{2}:\d{2}:\d{2}) - ([^:]+): (.*)$", line)
            if match:
                _, user, text = match.groups()
                text = text.rstrip().removesuffix('()').rstrip()
                timestamp = actual_timestamps[i] if i < len(actual_timestamps) else datetime.now()
                comment = {
                    'User Name': user,
                    'Text': text,
                    'Timestamp': timestamp
                }
                generated_comments.append(comment)
            else:
                generated_comments.append({'User Name': 'Unknown', 'Text': line, 'Timestamp': datetime.now()})
    # else: generated_comments stays empty

    col1, col2 = st.columns(2)
    if side_order == 'actual_left':
        left_label, right_label = "Conversation follow-up A", "Conversation follow-up B"
        left_comments, right_comments = actual_comments, generated_comments
        left_is_actual = True
    else:
        left_label, right_label = "Conversation follow-up A", "Conversation follow-up B"
        left_comments, right_comments = generated_comments, actual_comments
        left_is_actual = False

    with col1:
        st.subheader(left_label)
        for comment in left_comments:
            # If actual_comments, comment is (idx, row); if generated, it's dict
            if left_is_actual:
                display_comment(comment[1])
            else:
                display_comment(comment, is_placeholder=True)
    with col2:
        st.subheader(right_label)
        for comment in right_comments:
            if not left_is_actual:
                display_comment(comment[1])
            else:
                display_comment(comment, is_placeholder=True)

    # Annotation interface
    st.subheader("Annotation")
    slider_help = f"Move left for Conversation follow-up A, right for Conversation follow-up B. The further you move, the more confident you are."
    st.write(slider_help)
    
    # Get existing annotation if any
    existing_annotation = None
    if st.session_state.annotator_email in st.session_state.annotated_files:
        existing_annotation = st.session_state.annotated_files[st.session_state.annotator_email].get(str(current_file))
    
    # If there's an existing annotation, show it
    if existing_annotation:
        st.info("You have already annotated this conversation. Submitting a new annotation will replace the previous one.")
    
    # Set slider value: 0.0 for new thread, or existing annotation value
    slider_value = 0.0 if not existing_annotation else existing_annotation.get("confidence", 0.0)

    # Add confidence slider
    confidence = st.slider(
        "Confidence Score",
        min_value=-1.0,
        max_value=1.0,
        value=slider_value,
        step=0.1,
        key=f"confidence_slider_value_{str(current_file)}",
        help="Move left for comments in Conversation follow-up A (on the left), right for right comments in Conversation follow-up B (on the right). The further you move, the more confident you are."
    )
    
    # Add reasoning field
    reasoning = st.text_area(
        "(Optional) Reasoning for your choice",
        value="" if not existing_annotation else existing_annotation.get("reasoning", ""),
        height=150,
        help="Explain why you chose this comment and your confidence level"
    )
    
    # Add save button
    if st.button("Save Annotation"):
        # Initialize user's annotation dictionary if it doesn't exist
        if st.session_state.annotator_email not in st.session_state.annotated_files:
            st.session_state.annotated_files[st.session_state.annotator_email] = {}
        
        # Save new annotation
        if side_order == 'actual_left':
            left_comment_type = 'actual'
            right_comment_type = 'generated'
        else:
            left_comment_type = 'generated'
            right_comment_type = 'actual'
        st.session_state.annotated_files[st.session_state.annotator_email][str(current_file)] = {
            "confidence": confidence,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat(),
            "left_comment_type": left_comment_type,
            "right_comment_type": right_comment_type
        }
        
        # Save to file
        save_annotations()
        
        # Automatically load the next conversation
        next_conversation()

def next_conversation():
    """Move to the next conversation."""
    conversation_files = get_conversation_files()
    st.session_state.current_file_index = get_next_unannotated_file(conversation_files)
    st.experimental_rerun()

def load_statements():
    """Load statements from JSON file."""
    statements_path = "data/reddit/cmv/statements.json"
    with open(statements_path, 'r') as f:
        return json.load(f)

def load_annotations():
    """Load annotations from the user's annotations.json file."""
    if not st.session_state.annotator_email:
        st.session_state.annotated_files = {}
        return
    user_dir = get_user_dir(st.session_state.annotator_email)
    annotations_file = user_dir / 'annotations.json'
    try:
        with open(annotations_file, "r") as f:
            st.session_state.annotated_files = json.load(f)
    except FileNotFoundError:
        st.session_state.annotated_files = {}
    except json.JSONDecodeError:
        st.error("Error parsing annotations.json file!")
        st.session_state.annotated_files = {}

def save_annotations():
    """Save annotations to the user's annotations.json file."""
    if not st.session_state.annotator_email:
        return
    user_dir = get_user_dir(st.session_state.annotator_email)
    os.makedirs(user_dir, exist_ok=True)
    annotations_file = user_dir / 'annotations.json'
    with open(annotations_file, 'w') as f:
        json.dump(st.session_state.annotated_files, f, indent=2)

    try:
        upload_to_gcs(
            bucket_name=os.getenv('GCS_BUCKET_NAME'),
            source_file_name=annotations_file,
            destination_blob_name=f'{st.session_state.annotator_email}_annotations.json'
        )
        st.success("Annotation saved successfully!")
    except Exception as e:
        st.error(f"Failed to upload to cloud storage: {str(e)}")

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state variables
    if 'annotator_email' not in st.session_state:
        st.session_state.annotator_email = None
    if 'annotated_files' not in st.session_state:
        st.session_state.annotated_files = {}
    if 'current_file_index' not in st.session_state:
        st.session_state.current_file_index = 0
    if 'view' not in st.session_state:
        st.session_state.view = "login"
    if 'statements' not in st.session_state:
        st.session_state.statements = load_statements()
    
    # Load annotations
    load_annotations()
    
    # Show login page if not logged in
    if st.session_state.annotator_email is None:
        show_login_page()
        return

    # Only set current_file_index to next unannotated after login or explicit request
    if st.session_state.view != "list" and st.session_state.get('force_next_unannotated', False):
        conversation_files = get_conversation_files()
        st.session_state.current_file_index = get_next_unannotated_file(conversation_files)
        st.session_state.force_next_unannotated = False

    # Show conversation list if in list view
    if st.session_state.view == "list":
        show_conversation_list()
        return
    
    # Show annotation interface
    show_annotation_interface()

if __name__ == "__main__":
    main() 