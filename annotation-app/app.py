import streamlit as st
import pandas as pd
import os
from datetime import datetime
from google.cloud import storage
from pathlib import Path
import random

# Initialize session state
if 'current_file_index' not in st.session_state:
    st.session_state.current_file_index = 0
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

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
    st.markdown(f"""
    <div style="
        border-left: 4px solid {'#FF4500' if not is_placeholder else '#888'};
        padding-left: 10px;
        margin: 10px 0;
    ">
        <p style="color: #888; font-size: 0.8em;">
            u/{comment['User Name']} • {comment['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        </p>
        <p>{comment['Text']}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    st.title("Reddit Conversation Annotation Tool")
    
    # Get all conversation files
    conversation_files = get_conversation_files()
    
    if not conversation_files:
        st.error("No conversation files found!")
        return
    
    # Load current conversation
    current_file = conversation_files[st.session_state.current_file_index]
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
        'timestamp': timestamp
    }
    
    # Save locally first
    annotations_dir = Path(__file__).parent / 'annotations'
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_file = annotations_dir / f'annotation_{timestamp}.txt'
    with open(annotation_file, 'w') as f:
        f.write(str(annotation))
    
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
    st.session_state.current_file_index += 1
    st.session_state.processed_files.add(st.session_state.current_file_index - 1)
    st.experimental_rerun()

if __name__ == "__main__":
    main() 