# HearU - Mental Health Analysis Chat-Based Application

## Introduction

Mental health awareness has gained increasing importance in recent years. With advancements in natural language processing (NLP), AI-driven models now offer promising tools to assist users in understanding and managing their mental health. HearU leverages the LLaMA model to analyze user interactions for mental health indicators, providing insights that could support self-awareness, early intervention, and consistent tracking over time.

### Project Goals
The main goals of this project are:
1. Develop a chat-based application that enables seamless interaction for users who may be dealing with mental health issues.
2. Implement polarity detection, keyword extraction, concern classification, and intensity scoring of user inputs to generate timeline-based sentiment shift analysis reports.

## System Design and Architecture

### Overview
The application consists of two main components:
- **User Interface (UI)**: A context-aware, chat-based frontend powered by LLaMA.
- **Processing and Storage**: Responsible for NLP-based analysis, classifying, scoring, and tracking emotional patterns over time. Additionally, this component stores analysis for each user to retain context over time and across devices.

### Data Flow
1. **User Input**: User interactions are received in real-time via the chat interface.
2. **Identifier**: Classifies messages as potential signs of mental health concerns.
3. **Classifier**: Categorizes issues (e.g., depression, anxiety, stress).
4. **Scoring Module**: Assigns intensity scores based on sentiment analysis.
5. **Report Generation**: Collates scores, frequency, and category patterns into an evolving report, updated periodically for each user.

## Key Technologies

Key technologies used in the project include:

- **LLaMA Model**: An open-source NLP model designed for sophisticated language analysis and contextual understanding, known for its efficiency and flexibility.
- **Database Management**: Utilizes MongoDB or PostgreSQL for storing user interactions and generated reports.
- **Frontend Framework**: The React framework is used for a responsive and user-friendly chat interface.
- **Backend Framework**: Flask is employed to handle real-time communication, analysis requests, and data retrieval.

## Conclusion

This project represents a significant step toward proactive mental health management using AI. By combining advanced NLP with thoughtful design and analysis, HearU empowers users to monitor and understand their mental health over time, bridging the gap between awareness and action.

## Future Scope

While we have made HearU as complete and user-friendly as possible within the project's timeframe, there are additional features we would love to integrate to further enhance the system:

1. **Integration of Support Resources**: Recommending resources or helplines based on detected needs.
2. **Expansion to Multilingual Support**: Ensuring accessibility for non-English speakers.
3. **Offline Processing**: Performing processing even without an internet connection would be a significant enhancement.

## Dataset Resources

The following datasets are recommended and were used to train and test the application’s capabilities in mental health analysis:

- [Reddit Mental Health Data](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data)
- [Mental Health Corpus](https://www.kaggle.com/datasets/reihanenamdari/mental-health-corpus)
- [GoEmotions Dataset](https://github.com/google-research/google-research/tree/master/goemotions/data/full_dataset)
- [Mental Health Sentiment Analysis - NLP ML](https://www.kaggle.com/code/annastasy/mental-health-sentiment-analysis-nlp-ml)
- [National Institute of Mental Health Statistics](https://www.nimh.nih.gov/health/statistics/mental-illness)
- [Sentiment Classification (arXiv)](https://arxiv.org/abs/1802.08379)

<hr>
<hr>
<hr>