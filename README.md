# Misson Gemini Dynamo

DynamoCards is an open-source tool designed to simplify the task of parsing lengthy YouTube transcripts using its advanced Semantic Extraction Algorithm (SEA). This tool is particularly beneficial for students and educators, as it helps streamline the study process by efficiently identifying and organizing key concepts and terms within university lectures and other lengthy video content.

## Key Features

- **Semantic Extraction Algorithm (SEA)**: Efficiently parses lengthy YouTube transcripts to extract and organize key concepts.
- **Educational Enhancement**: Facilitates more effective study habits and enhances classroom instruction by condensing hours of lecture material into concise, digestible insights.
- **User-Friendly Interface**: The frontend, built with React, allows users to input a YouTube URL and receive organized keyword flashcards as output.
- **Advanced Backend Processing**: Utilizes YouTube API, GCP Vertex AI Gemini model, and custom processing objects to handle and analyze video transcripts.

## System Overview

1. **Frontend**:

   - Built with React.
   - Takes a YouTube URL as input and sends a request to the backend.

2. **Backend**:

   - Receives the request and uses the `YoutubeProcessor` object to fetch the transcript using the YouTube API.
   - Generates documents from the video transcript.
   - Processes the documents into a custom amount of groups.
   - Each group of documents is processed by the `Geminiprocessor` object using the GCP Vertex AI Gemini model, including a cost analysis.
   - Formats the results from different groups into a JSON object and stacks them into a list.
   - Sends the JSON object list back to the frontend.

3. **Output**:
   - The frontend displays the processed data as keyword flashcards.
