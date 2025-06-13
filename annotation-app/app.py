import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime
from google.cloud import storage
from pathlib import Path

# Initialize session state
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()
if 'annotator_email' not in st.session_state:
    st.session_state.annotator_email = None
if 'statements' not in st.session_state:
    # Load statements from JSON file
    parent_dir = Path(__file__).parent.parent
    statements_path = parent_dir / "data/reddit/cmv/statements.json"
    with open(statements_path, 'r') as f:
        st.session_state.statements = json.load(f)
if 'annotated_files' not in st.session_state:
    st.session_state.annotated_files = {}

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
    # Get the parent directory of annotation-app
    parent_dir = Path(__file__).parent.parent
    data_dir = parent_dir / "data/reddit/cmv/test"
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
    st.markdown(footer_html, unsafe_allow_html=True)

def load_annotated_files():
    """Load the annotated files for all users."""
    annotations_dir = Path(__file__).parent / 'annotations'
    progress_file = annotations_dir / 'annotation_progress.json'
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {}

def save_annotated_files():
    """Save the annotated files for all users."""
    annotations_dir = Path(__file__).parent / 'annotations'
    os.makedirs(annotations_dir, exist_ok=True)
    progress_file = annotations_dir / 'annotation_progress.json'
    with open(progress_file, 'w') as f:
        json.dump(st.session_state.annotated_files, f)

def get_next_unannotated_file(conversation_files):
    """Get the next file that hasn't been annotated by the current user."""
    if st.session_state.annotator_email not in st.session_state.annotated_files:
        st.session_state.annotated_files[st.session_state.annotator_email] = []
    
    annotated_files = st.session_state.annotated_files[st.session_state.annotator_email]
    for i, file in enumerate(conversation_files):
        if str(file) not in annotated_files:
            return i
    return 0  # If all files are annotated, start from the beginning

def show_login_page():
    """Display the login page with email input."""
    st.title("Reddit Conversation Annotation Tool")
    st.write("Please enter your email address to begin annotation.")
    
    email = st.text_input("Email Address")
    if st.button("Start Annotation"):
        if email and "@" in email and "." in email.split("@")[1]:  # Better email validation
            st.session_state.annotator_email = email
            # Load annotated files when user logs in
            st.session_state.annotated_files = load_annotated_files()
            # Set current file index to next unannotated file
            conversation_files = get_conversation_files()
            st.session_state.current_file_index = get_next_unannotated_file(conversation_files)
            st.experimental_rerun()
        else:
            st.error("Please enter a valid email address (e.g., user@example.com)")

def show_conversation_list():
    """Display a list of all conversations with their annotation status."""
    st.title("All Conversations")
    
    # Display annotator info and progress
    st.sidebar.write(f"Annotator: {st.session_state.annotator_email}")
    conversation_files = get_conversation_files()
    annotated_count = len(st.session_state.annotated_files.get(st.session_state.annotator_email, []))
    total_count = len(conversation_files)
    st.sidebar.write(f"Progress: {annotated_count}/{total_count} conversations annotated")
    
    # Add navigation buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    
    if st.sidebar.button("Back to Annotation"):
        st.session_state.view = "annotation"
        st.experimental_rerun()
    
    if st.sidebar.button("Change Email"):
        st.session_state.annotator_email = None
        st.experimental_rerun()
    
    # Get all conversations
    if not conversation_files:
        st.error("No conversation files found!")
        return
    
    # Create a search box
    search_query = st.text_input("Search conversations", "")
    
    # Get annotated files for current user
    annotated_files = st.session_state.annotated_files.get(st.session_state.annotator_email, [])
    
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
        col2.markdown(f"**{title}**")
        col2.markdown(f"*Submission ID:* {conversation_id}  \n*Thread ID:* {thread_id}")
        
        # View button
        if col3.button("View", key=f"view_{i}"):
            # Find the index of this conversation in the full list
            for j, f in enumerate(conversation_files):
                if f == file:
                    st.session_state.current_file_index = j
                    st.session_state.view = "annotation"
                    st.experimental_rerun()
                    break

def show_annotation_interface():
    """Display the main annotation interface."""
    st.title("Reddit Conversation Annotation Tool")
    
    # Display annotator info and progress
    st.sidebar.write(f"Annotator: {st.session_state.annotator_email}")
    conversation_files = get_conversation_files()
    annotated_count = len(st.session_state.annotated_files.get(st.session_state.annotator_email, []))
    total_count = len(conversation_files)
    st.sidebar.write(f"Progress: {annotated_count}/{total_count} conversations annotated")
    
    # Add navigation buttons in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Navigation")
    if st.sidebar.button("View All Conversations"):
        st.session_state.view = "list"
        st.experimental_rerun()
    
    if st.sidebar.button("Change Email"):
        st.session_state.annotator_email = None
        st.experimental_rerun()
    
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
    st.markdown(f'**Submission ID:** {conversation_id} | **Thread ID:** {thread_id}')
    
    df = load_conversation(current_file)
    
    # Display conversation history (excluding last 2 comments)
    st.subheader("Conversation History")
    for _, comment in df[:-2].iterrows():
        display_comment(comment)
    
    # Create two columns for the last two comments and placeholders
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actual Comments")
        for _, comment in df[-2:].iterrows():
            display_comment(comment)
    
    with col2:
        st.subheader("Placeholder Comments")
        # Create placeholder comments with similar structure
        for _ in range(2):
            placeholder = {
                'User Name': 'PlaceholderUser',
                'Text': 'This is a placeholder comment.',
                'Timestamp': datetime.now()
            }
            display_comment(placeholder, is_placeholder=True)
    
    # Annotation interface
    st.subheader("Annotation")
    st.write("Select the pair of comments that were written by humans:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Left Pair (Actual Comments)"):
            save_annotation(current_file, "left")
            next_conversation()
    
    with col2:
        if st.button("Right Pair (Placeholder Comments)"):
            save_annotation(current_file, "right")
            next_conversation()

def save_annotation(file_path, selection):
    """Save the annotation to a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    annotation = {
        'file': str(file_path),
        'selection': selection,
        'timestamp': timestamp,
        'annotator': st.session_state.annotator_email
    }
    
    # Save locally first
    annotations_dir = Path(__file__).parent / 'annotations'
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_file = annotations_dir / f'annotation_{timestamp}.txt'
    with open(annotation_file, 'w') as f:
        f.write(str(annotation))
    
    # Add to annotated files for this user
    if st.session_state.annotator_email not in st.session_state.annotated_files:
        st.session_state.annotated_files[st.session_state.annotator_email] = []
    st.session_state.annotated_files[st.session_state.annotator_email].append(str(file_path))
    save_annotated_files()
    
    # Upload to GCS
    try:
        upload_to_gcs(
            bucket_name=os.getenv('GCS_BUCKET_NAME'),
            source_file_name=str(annotation_file),
            destination_blob_name=f'annotations/annotation_{timestamp}.txt'
        )
        st.success("Annotation saved successfully!")
    except Exception as e:
        st.error(f"Failed to upload to cloud storage: {str(e)}")

def next_conversation():
    """Move to the next conversation."""
    conversation_files = get_conversation_files()
    st.session_state.current_file_index = get_next_unannotated_file(conversation_files)
    st.experimental_rerun()

def main():
    # Initialize view state
    if 'view' not in st.session_state:
        st.session_state.view = "annotation"
    
    if st.session_state.annotator_email is None:
        show_login_page()
    else:
        if st.session_state.view == "list":
            show_conversation_list()
        else:
            show_annotation_interface()

if __name__ == "__main__":
    main() 