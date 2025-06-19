# Reddit Conversation Annotation Tool

A Streamlit application for annotating Reddit conversations by identifying which pair of comments were written by humans.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Google Cloud Storage:
   - Create a Google Cloud Storage bucket
   - Create a `.env` file with the following content:
   ```
   GCS_BUCKET_NAME=your-bucket-name
   ```
   - Set up Google Cloud credentials (either through environment variable GOOGLE_APPLICATION_CREDENTIALS or by running `gcloud auth application-default login`)

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. The application will:
   - Display Reddit conversations
   - Show the conversation history (excluding the last two comments)
   - Present the last two actual comments on the left
   - Show placeholder comments on the right
   - Allow you to select which pair was written by humans
   - Save annotations both locally and to Google Cloud Storage
   - Move to the next conversation automatically after annotation

## Features

- Reddit-like comment display format
- Automatic conversation progression
- Cloud storage integration
- Local backup of annotations
- Session state management to track progress

## Directory Structure

```
.
├── app.py
├── requirements.txt
├── README.md
└── annotations/  (created automatically)
```

# TODO swap 1st and 2nd comment could be after same source comment or not
# load generated comments
# Google Cloud Bucket
# 45adoi ill rendered