from llama_cpp import Llama
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Optional, Dict
import sqlite3
import json
from datetime import datetime, timedelta
import subprocess
import asyncio
from pathlib import Path
import os
import torch
import spacy
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# FastAPI Models
class User(BaseModel):
    user_id: str
    name: Optional[str] = None


class Message(BaseModel):
    text: str


class ConversationHistory(BaseModel):
    messages: List[Dict]
    analysis: Optional[Dict] = None


class MentalHealthCategories:
    """Class to define mental health categories and associated keywords"""
    CATEGORIES = {
        'anxiety': [
            # Core symptoms
            'anxiety', 'panic', 'worry', 'nervousness', 'restlessness', 'unease',
            'fear', 'dread', 'apprehension', 'tension', 'stress',

            # Physical symptoms
            'racing heart', 'sweating', 'trembling', 'shaking', 'chest pain',
            'breathing difficulty', 'shortness of breath', 'dizziness', 'lightheaded',
            'nausea', 'stomach upset',

            # Mental symptoms
            'racing thoughts', 'overthinking', 'rumination', 'mind racing',
            'can\'t stop thinking', 'intrusive thoughts', 'worried thoughts',
            'catastrophizing', 'anticipatory anxiety',

            # Types of anxiety
            'generalized anxiety', 'panic attacks', 'anxiety attacks',
            'acute anxiety', 'chronic anxiety', 'severe anxiety', 'mild anxiety',

            # Impact and experience
            'overwhelming', 'debilitating', 'paralyzing', 'crippling',
            'constant worry', 'always anxious', 'anxiety triggers',
            'anxiety symptoms', 'anxiety management'
        ],

        'depression': [
            # Core symptoms
            'depression', 'sadness', 'hopelessness', 'despair', 'emptiness',
            'numbness', 'apathy', 'melancholy', 'gloom', 'misery',

            # Emotional symptoms
            'feeling down', 'low mood', 'mood swings', 'emotional pain',
            'emotional numbness', 'emotional exhaustion', 'crying',
            'tearfulness', 'grief-like', 'darkness',

            # Mental symptoms
            'negative thoughts', 'self-hatred', 'worthlessness', 'guilt',
            'shame', 'regret', 'self-criticism', 'hopeless thoughts',
            'suicidal thoughts', 'death thoughts',

            # Physical symptoms
            'fatigue', 'exhaustion', 'low energy', 'lethargy', 'insomnia',
            'oversleeping', 'appetite changes', 'weight changes',
            'physical symptoms', 'body aches',

            # Behavioral symptoms
            'withdrawal', 'isolation', 'avoiding people', 'loss of interest',
            'no motivation', 'can\'t enjoy things', 'stopped activities',
            'procrastination', 'neglecting responsibilities'
        ],

        'academic_stress': [
            # Core concerns
            'academic stress', 'study pressure', 'school stress', 'exam stress',
            'academic pressure', 'academic anxiety', 'academic worry',

            # Specific stressors
            'exams', 'tests', 'quizzes', 'assignments', 'homework',
            'projects', 'presentations', 'deadlines', 'grades', 'GPA',
            'academic performance', 'academic achievement',

            # Study-related
            'studying', 'cramming', 'revision', 'preparation',
            'concentration', 'focus', 'memory', 'understanding',
            'learning difficulties', 'study habits',

            # Academic environment
            'classroom anxiety', 'lecture stress', 'university pressure',
            'college stress', 'school environment', 'academic competition',
            'peer pressure', 'teacher pressure', 'parental expectations',

            # Impact
            'academic burnout', 'study fatigue', 'mental exhaustion',
            'academic overwhelm', 'academic frustration', 'academic fear',
            'academic failure', 'academic success', 'academic goals'
        ],

        'relationship_issues': [
            # General relationship concerns
            'relationship problems', 'relationship stress', 'relationship anxiety',
            'relationship fears', 'relationship doubts', 'relationship confusion',

            # Specific relationships
            'romantic relationships', 'dating issues', 'marriage problems',
            'partnership issues', 'friendship problems', 'work relationships',

            # Common problems
            'trust issues', 'communication problems', 'commitment issues',
            'intimacy problems', 'compatibility issues', 'relationship conflict',
            'jealousy', 'insecurity', 'attachment issues',

            # Relationship events
            'breakup', 'divorce', 'separation', 'cheating', 'infidelity',
            'betrayal', 'reconciliation', 'relationship changes',

            # Relationship patterns
            'toxic relationships', 'unhealthy relationships', 'codependency',
            'relationship patterns', 'relationship trauma', 'relationship habits',
            'relationship expectations', 'relationship boundaries'
        ],

        'career_confusion': [
            # Core concerns
            'career confusion', 'job uncertainty', 'career uncertainty',
            'career indecision', 'career doubt', 'professional confusion',

            # Career development
            'career path', 'career direction', 'career planning',
            'career goals', 'career growth', 'career development',
            'career progression', 'career advancement',

            # Job-related
            'job search', 'job applications', 'interviews', 'job hunting',
            'job market', 'employment', 'unemployment', 'job security',

            # Career changes
            'career change', 'career transition', 'career switch',
            'new career', 'career shift', 'career move',

            # Professional identity
            'professional identity', 'career identity', 'professional goals',
            'career satisfaction', 'work fulfillment', 'career passion',
            'dream job', 'career fit'
        ]
    }

    @classmethod
    def get_all_categories(cls):
        return list(cls.CATEGORIES.keys())

    @classmethod
    def get_keywords(cls, category):
        return cls.CATEGORIES.get(category, [])


class MentalHealthAnalyzer:
    """Core analyzer class for mental health text analysis"""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except IOError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.intensity_scaler = MinMaxScaler()
        self.timeline_data = []
        self.categories = MentalHealthCategories.get_all_categories()

    def train(self, training_data):
        texts = [item['text'] for item in training_data]
        categories = [item['category'] for item in training_data]
        intensities = [item['intensity'] for item in training_data]

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, categories)
        self.intensity_scaler.fit(np.array(intensities).reshape(-1, 1))

    def analyze_text(self, text, timestamp=None):
        text = text.strip()
        timestamp = timestamp or datetime.now()
        timestamp_str = timestamp.isoformat()  # Convert to ISO format string

        # Sentiment Analysis
        sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
        sentiment = self._get_sentiment_label(sentiment_scores['compound'])

        # Category Classification
        X = self.vectorizer.transform([text])
        category = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]

        # Keywords Extraction
        keywords = self.extract_keywords(text)

        # Intensity Scoring
        intensity = self._calculate_intensity(text, sentiment_scores)

        result = {
            'text': text,
            'timestamp': timestamp_str,  # Use the string version
            'sentiment': {
                'label': sentiment,
                'scores': sentiment_scores
            },
            'category': {
                'primary': category,
                'confidence': float(max(probs)),
                # Convert numpy array to list
                'all_probabilities': dict(zip(self.classifier.classes_, probs.tolist()))
            },
            'keywords': keywords,
            'intensity': intensity
        }

        self.timeline_data.append(result)
        return result

    def _get_sentiment_label(self, compound_score):
        if compound_score >= 0.05:
            return 'Positive'
        if compound_score <= -0.05:
            return 'Negative'
        return 'Neutral'

    def extract_keywords(self, text):
        doc = self.nlp(text)
        keywords = []

        # NER
        for ent in doc.ents:
            keywords.append({
                'text': ent.text,
                'type': 'named_entity',
                'label': ent.label_
            })

        # Mental health terms
        for category, terms in MentalHealthCategories.CATEGORIES.items():
            for term in terms:
                if term in text.lower():
                    keywords.append({
                        'text': term,
                        'type': 'mental_health_term',
                        'category': category
                    })

        return keywords

    def _calculate_intensity(self, text, sentiment_scores):
        intensity_words = {
            'high': {'extremely', 'severe', 'very', 'always', 'completely'},
            'medium': {'often', 'somewhat', 'moderate', 'sometimes'},
            'low': {'slightly', 'mild', 'occasionally', 'rarely'}
        }

        words = text.lower().split()
        high_count = sum(word in intensity_words['high'] for word in words)
        medium_count = sum(word in intensity_words['medium'] for word in words)
        low_count = sum(word in intensity_words['low'] for word in words)

        base_score = (high_count * 3 + medium_count *
                      2 + low_count) / max(len(words), 1)
        sentiment_impact = abs(sentiment_scores['compound'])

        final_score = (base_score + sentiment_impact) / 2 * 10
        final_score = max(min(final_score, 10), 1)

        return {
            'score': round(final_score, 1),
            'level': 'High' if final_score >= 7 else 'Medium' if final_score >= 4 else 'Low'
        }


class DatabaseManager:
    def __init__(self, db_path="mental_health.db"):
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # Update users table to use user_id
        c.execute('''
            CREATE TABLE IF NOT EXISTS users
            (user_id TEXT PRIMARY KEY, name TEXT)
        ''')

        # Update conversations table to use user_id
        c.execute('''
            CREATE TABLE IF NOT EXISTS conversations
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
             user_id TEXT,
             message TEXT,
             timestamp TEXT,
             analysis TEXT,
             model_response TEXT)
        ''')

        conn.commit()
        conn.close()

    def add_user(self, user_id: str, name: Optional[str] = None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('INSERT OR REPLACE INTO users (user_id, name) VALUES (?, ?)',
                  (user_id, name))
        conn.commit()
        conn.close()

    def add_conversation(self, user_id: str, message: str, analysis: dict, model_response: str):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        timestamp = datetime.now().isoformat()
        c.execute('''
            INSERT INTO conversations 
            (user_id, message, timestamp, analysis, model_response)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, message, timestamp, json.dumps(analysis), model_response))
        conn.commit()
        conn.close()

    def get_conversation_history(self, user_id: str) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            SELECT message, timestamp, analysis, model_response 
            FROM conversations 
            WHERE user_id = ? 
            ORDER BY timestamp
        ''', (user_id,))
        rows = c.fetchall()
        conn.close()

        return [
            {
                'message': row[0],
                'timestamp': row[1],
                'analysis': json.loads(row[2]),
                'model_response': row[3]
            }
            for row in rows
        ]


class LlamaManager:
    def __init__(self):
        try:
            self.llm = Llama.from_pretrained(
                repo_id="sujal011/llama3.2-3b-mental-health-chatbot",
                filename="unsloth.Q8_0.gguf",
            )
            self.conversation_histories = {}
        except Exception as e:
            logger.error(f"Failed to initialize Llama model: {str(e)}")
            raise

    async def start_conversation(self, user_id: str):
        if user_id not in self.conversation_histories:
            self.conversation_histories[user_id] = []
            logger.info(f"Started conversation for user {user_id}")

    async def get_response(self, user_id: str, message: str) -> str:
        if user_id not in self.conversation_histories:
            await self.start_conversation(user_id)

        conversation = self.conversation_histories[user_id]
        conversation.append({"role": "user", "content": message})

        try:
            response = self.llm.create_chat_completion(
                messages=conversation
            )
            assistant_message = response['choices'][0]['message']['content']
            conversation.append({
                "role": "assistant",
                "content": assistant_message
            })
            return assistant_message
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while processing your message. Please try again."

    def close_conversation(self, user_id: str):
        if user_id in self.conversation_histories:
            del self.conversation_histories[user_id]
            logger.info(f"Closed conversation for user {user_id}")


class MentalHealthVisualizer:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def plot_sentiment_timeline(self):
        timeline_data = pd.DataFrame(self.analyzer.timeline_data)
        timeline_data['timestamp'] = pd.to_datetime(timeline_data['timestamp'])
        timeline_data['sentiment_score'] = timeline_data['sentiment'].apply(
            lambda x: x['scores']['compound']
        )

        plt.figure(figsize=(12, 6))
        plt.plot(timeline_data['timestamp'],
                 timeline_data['sentiment_score'], 'b-')
        plt.title('Sentiment Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sentiment Score')
        plt.grid(True)
        plt.show()

    def plot_category_distribution(self):
        categories = [entry['category']['primary']
                      for entry in self.analyzer.timeline_data]
        category_counts = pd.Series(categories).value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Distribution of Mental Health Categories')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def generate_interactive_dashboard(self):
        df = pd.DataFrame(self.analyzer.timeline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: x['scores']['compound'])
        df['intensity_score'] = df['intensity'].apply(lambda x: x['score'])
        df['category'] = df['category'].apply(lambda x: x['primary'])

        # Sentiment Timeline
        fig1 = px.line(df, x='timestamp', y='sentiment_score',
                       title='Sentiment Trend Over Time')
        fig1.show()

        # Category Distribution
        fig2 = px.bar(df['category'].value_counts(),
                      title='Category Distribution')
        fig2.show()

        # Intensity Scatter
        fig3 = px.scatter(df, x='timestamp', y='intensity_score',
                          color='category', size='intensity_score',
                          title='Intensity Distribution Over Time')
        fig3.show()


# Initialize FastAPI app
app = FastAPI(title="Mental Health Analysis System")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize core components
db_manager = DatabaseManager()
model_manager = LlamaManager()


class MentalHealthAnalysisSystem:
    def __init__(self):
        self.analyzer = MentalHealthAnalyzer()
        self.visualizer = MentalHealthVisualizer(self.analyzer)
        self.training_data = self._get_training_data()
        self.train_system()

    def _get_training_data(self):
        return [
            # Anxiety (General)
            {
                'text': 'I feel constantly worried about everything and can\'t control my thoughts',
                'polarity': 'Negative',
                'extracted_concern': 'constant worry',
                'category': 'anxiety',
                'intensity': 8,
                'keywords': ['worried', 'constantly', 'control']
            },
            {
                'text': 'My mind races with anxious thoughts, especially at night',
                'polarity': 'Negative',
                'extracted_concern': 'racing thoughts',
                'category': 'anxiety',
                'intensity': 7,
                'keywords': ['racing thoughts', 'anxious', 'night']
            },
            {
                'text': 'Starting to feel better about managing my anxiety with meditation',
                'polarity': 'Positive',
                'extracted_concern': 'improving anxiety',
                'category': 'anxiety',
                'intensity': 4,
                'keywords': ['better', 'anxiety', 'meditation']
            },
            {
                'text': 'Panic attacks are becoming more frequent and intense',
                'polarity': 'Negative',
                'extracted_concern': 'panic attacks',
                'category': 'anxiety',
                'intensity': 9,
                'keywords': ['panic attacks', 'frequent', 'intense']
            },
            {
                'text': 'Feeling anxious about social situations but trying exposure therapy',
                'polarity': 'Neutral',
                'extracted_concern': 'social anxiety treatment',
                'category': 'anxiety',
                'intensity': 6,
                'keywords': ['anxious', 'social', 'therapy']
            },

            # Depression
            {
                'text': 'Nothing brings me joy anymore, feeling completely empty inside',
                'polarity': 'Negative',
                'extracted_concern': 'anhedonia',
                'category': 'depression',
                'intensity': 9,
                'keywords': ['joy', 'empty', 'nothing']
            },
            {
                'text': 'Can\'t get out of bed most days, everything feels overwhelming',
                'polarity': 'Negative',
                'extracted_concern': 'lack of motivation',
                'category': 'depression',
                'intensity': 8,
                'keywords': ['bed', 'overwhelming']
            },
            {
                'text': 'Started antidepressants, feeling slightly more energetic',
                'polarity': 'Positive',
                'extracted_concern': 'treatment progress',
                'category': 'depression',
                'intensity': 5,
                'keywords': ['antidepressants', 'energetic']
            },
            {
                'text': 'Depression making it hard to concentrate at work',
                'polarity': 'Negative',
                'extracted_concern': 'cognitive issues',
                'category': 'depression',
                'intensity': 7,
                'keywords': ['depression', 'concentrate', 'work']
            },
            {
                'text': 'Feeling hopeless about the future despite therapy',
                'polarity': 'Negative',
                'extracted_concern': 'hopelessness',
                'category': 'depression',
                'intensity': 8,
                'keywords': ['hopeless', 'future', 'therapy']
            },

            # Academic Stress
            {
                'text': 'Overwhelmed by upcoming finals and assignment deadlines',
                'polarity': 'Negative',
                'extracted_concern': 'academic pressure',
                'category': 'academic_stress',
                'intensity': 8,
                'keywords': ['finals', 'deadlines', 'overwhelmed']
            },
            {
                'text': 'Can\'t focus on studying, worried about failing',
                'polarity': 'Negative',
                'extracted_concern': 'academic performance',
                'category': 'academic_stress',
                'intensity': 7,
                'keywords': ['studying', 'failing', 'focus']
            },
            {
                'text': 'Created a study schedule, feeling more organized',
                'polarity': 'Positive',
                'extracted_concern': 'academic management',
                'category': 'academic_stress',
                'intensity': 4,
                'keywords': ['study', 'schedule', 'organized']
            },
            {
                'text': 'Pressure to maintain high GPA is affecting my sleep',
                'polarity': 'Negative',
                'extracted_concern': 'academic pressure',
                'category': 'academic_stress',
                'intensity': 7,
                'keywords': ['GPA', 'pressure', 'sleep']
            },
            {
                'text': 'Worried about college applications and future prospects',
                'polarity': 'Negative',
                'extracted_concern': 'future academics',
                'category': 'academic_stress',
                'intensity': 6,
                'keywords': ['college', 'applications', 'future']
            },

            # Relationship Issues
            {
                'text': 'Having constant arguments with my partner about trust',
                'polarity': 'Negative',
                'extracted_concern': 'relationship conflict',
                'category': 'relationship_issues',
                'intensity': 8,
                'keywords': ['arguments', 'partner', 'trust']
            },
            {
                'text': 'Feeling disconnected from my family after moving away',
                'polarity': 'Negative',
                'extracted_concern': 'family distance',
                'category': 'relationship_issues',
                'intensity': 6,
                'keywords': ['disconnected', 'family', 'moving']
            },
            {
                'text': 'Communication with spouse has improved after counseling',
                'polarity': 'Positive',
                'extracted_concern': 'relationship improvement',
                'category': 'relationship_issues',
                'intensity': 4,
                'keywords': ['communication', 'spouse', 'counseling']
            },
            {
                'text': 'Struggling to maintain friendships due to depression',
                'polarity': 'Negative',
                'extracted_concern': 'social relationships',
                'category': 'relationship_issues',
                'intensity': 7,
                'keywords': ['friendships', 'struggling', 'depression']
            },
            {
                'text': 'Trust issues from past relationship affecting current one',
                'polarity': 'Negative',
                'extracted_concern': 'trust issues',
                'category': 'relationship_issues',
                'intensity': 7,
                'keywords': ['trust', 'relationship', 'past']
            },

            # Career Confusion
            {
                'text': 'Unsure about my career path, feeling lost',
                'polarity': 'Negative',
                'extracted_concern': 'career uncertainty',
                'category': 'career_confusion',
                'intensity': 6,
                'keywords': ['career', 'lost', 'unsure']
            },
            {
                'text': 'Considering a career change but scared of the unknown',
                'polarity': 'Negative',
                'extracted_concern': 'career transition',
                'category': 'career_confusion',
                'intensity': 7,
                'keywords': ['career change', 'scared', 'unknown']
            },
            {
                'text': 'Started career counseling to explore options',
                'polarity': 'Positive',
                'extracted_concern': 'career exploration',
                'category': 'career_confusion',
                'intensity': 4,
                'keywords': ['career counseling', 'options']
            },
            {
                'text': 'Feeling stuck in current job but afraid to leave',
                'polarity': 'Negative',
                'extracted_concern': 'career stagnation',
                'category': 'career_confusion',
                'intensity': 6,
                'keywords': ['stuck', 'job', 'afraid']
            },
            {
                'text': 'Worried my skills aren\'t relevant for the job market',
                'polarity': 'Negative',
                'extracted_concern': 'career skills',
                'category': 'career_confusion',
                'intensity': 7,
                'keywords': ['skills', 'job market', 'worried']
            },

            # Health Anxiety
            {
                'text': 'Constantly googling symptoms and fearing the worst',
                'polarity': 'Negative',
                'extracted_concern': 'health anxiety',
                'category': 'health_anxiety',
                'intensity': 8,
                'keywords': ['symptoms', 'googling', 'worst']
            },
            {
                'text': 'Every small body sensation makes me panic about serious illness',
                'polarity': 'Negative',
                'extracted_concern': 'health paranoia',
                'category': 'health_anxiety',
                'intensity': 9,
                'keywords': ['body', 'panic', 'illness']
            },
            {
                'text': 'Working with therapist to manage health-related fears',
                'polarity': 'Neutral',
                'extracted_concern': 'health anxiety management',
                'category': 'health_anxiety',
                'intensity': 5,
                'keywords': ['therapist', 'health', 'fears']
            },
            {
                'text': 'Obsessively checking my pulse and blood pressure',
                'polarity': 'Negative',
                'extracted_concern': 'health monitoring',
                'category': 'health_anxiety',
                'intensity': 8,
                'keywords': ['pulse', 'blood pressure', 'checking']
            },
            {
                'text': 'Fear of getting sick is preventing me from going out',
                'polarity': 'Negative',
                'extracted_concern': 'illness anxiety',
                'category': 'health_anxiety',
                'intensity': 7,
                'keywords': ['sick', 'fear', 'going out']
            },

            # Social Anxiety
            {
                'text': 'Terrified of speaking up in meetings at work',
                'polarity': 'Negative',
                'extracted_concern': 'social fear',
                'category': 'social_anxiety',
                'intensity': 8,
                'keywords': ['speaking', 'meetings', 'terrified']
            },
            {
                'text': 'Avoiding social gatherings due to anxiety',
                'polarity': 'Negative',
                'extracted_concern': 'social avoidance',
                'category': 'social_anxiety',
                'intensity': 7,
                'keywords': ['avoiding', 'social', 'gatherings']
            },
            {
                'text': 'Made progress in group therapy for social anxiety',
                'polarity': 'Positive',
                'extracted_concern': 'treatment progress',
                'category': 'social_anxiety',
                'intensity': 4,
                'keywords': ['progress', 'group therapy', 'social']
            },
            {
                'text': 'Constant fear of judgment from others',
                'polarity': 'Negative',
                'extracted_concern': 'social judgment',
                'category': 'social_anxiety',
                'intensity': 8,
                'keywords': ['fear', 'judgment', 'others']
            },
            {
                'text': 'Heart races when having to introduce myself to new people',
                'polarity': 'Negative',
                'extracted_concern': 'social interaction',
                'category': 'social_anxiety',
                'intensity': 7,
                'keywords': ['heart races', 'introduce', 'new people']
            },

            # Eating Disorders
            {
                'text': 'Feeling guilty and anxious after every meal',
                'polarity': 'Negative',
                'extracted_concern': 'food guilt',
                'category': 'eating_disorder',
                'intensity': 8,
                'keywords': ['guilty', 'anxious', 'meal']
            },
            {
                'text': 'Started following a recovery meal plan with nutritionist',
                'polarity': 'Positive',
                'extracted_concern': 'recovery process',
                'category': 'eating_disorder',
                'intensity': 5,
                'keywords': ['recovery', 'meal plan', 'nutritionist']
            },
            {
                'text': 'Obsessing over calories and weight constantly',
                'polarity': 'Negative',
                'extracted_concern': 'food obsession',
                'category': 'eating_disorder',
                'intensity': 9,
                'keywords': ['calories', 'weight', 'obsessing']
            },
            {
                'text': 'Body image issues affecting my daily life',
                'polarity': 'Negative',
                'extracted_concern': 'body image',
                'category': 'eating_disorder',
                'intensity': 8,
                'keywords': ['body image', 'affecting', 'daily']
            },
            {
                'text': 'Fear of eating in public places',
                'polarity': 'Negative',
                'extracted_concern': 'social eating',
                'category': 'eating_disorder',
                'intensity': 7,
                'keywords': ['eating', 'public', 'fear']
            },

            # Insomnia
            {
                'text': 'Haven\'t had a full night\'s sleep in weeks',
                'polarity': 'Negative',
                'extracted_concern': 'sleep deprivation',
                'category': 'insomnia',
                'intensity': 9,
                'keywords': ['sleep', 'weeks', 'full night']
            },
            {
                'text': 'Mind races when trying to fall asleep',
                'polarity': 'Negative',
                'extracted_concern': 'sleep difficulty',
                'category': 'insomnia',
                'intensity': 7,
                'keywords': ['mind races', 'fall asleep']
            },
            {
                'text': 'Sleep medication helping but worried about dependency',
                'polarity': 'Neutral',
                'extracted_concern': 'sleep treatment',
                'category': 'insomnia',
                'intensity': 6,
                'keywords': ['sleep medication', 'dependency', 'worried']
            },
            {
                'text': 'Trying CBT-I techniques for better sleep',
                'polarity': 'Positive',
                'extracted_concern': 'sleep improvement',
                'category': 'insomnia',
                'intensity': 5,
                'keywords': ['CBT-I', 'techniques', 'sleep']
            },
            {
                'text': 'Irregular work schedule destroying my sleep pattern',
                'polarity': 'Negative',
                'extracted_concern': 'sleep disruption',
                'category': 'insomnia',
                'intensity': 8,
                'keywords': ['work schedule', 'sleep pattern', 'irregular']
            },

            # Self-Esteem
            {
                'text': 'Feeling worthless and inadequate compared to peers',
                'polarity': 'Negative',
                'extracted_concern': 'low self-worth',
                'category': 'self_esteem',
                'intensity': 8,
                'keywords': ['worthless', 'inadequate', 'peers']
            },
            {
                'text': 'Starting to recognize my own achievements',
                'polarity': 'Positive',
                'extracted_concern': 'self-improvement',
                'category': 'self_esteem',
                'intensity': 4,
                'keywords': ['achievements', 'recognize']
            },
            {
                'text': 'Constant self-criticism affecting my confidence',
                'polarity': 'Negative',
                'extracted_concern': 'self-criticism',
                'category': 'self_esteem',
                'intensity': 7,
                'keywords': ['self-criticism', 'confidence']
            },
            {
                'text': 'Difficulty accepting compliments or praise',
                'polarity': 'Negative',
                'extracted_concern': 'self-acceptance',
                'category': 'self_esteem',
                'intensity': 6,
                'keywords': ['compliments', 'praise', 'difficulty']
            },
            {
                'text': 'Working on self-compassion with my therapist',
                'polarity': 'Positive',
                'extracted_concern': 'self-improvement',
                'category': 'self_esteem',
                'intensity': 5,
                'keywords': ['self-compassion', 'therapist', 'working']
            },

            # Financial Stress
            {
                'text': 'Overwhelming debt keeping me up at night',
                'polarity': 'Negative',
                'extracted_concern': 'debt anxiety',
                'category': 'financial_stress',
                'intensity': 9,
                'keywords': ['debt', 'overwhelming', 'night']
            },
            {
                'text': 'Created a budget plan, feeling more in control',
                'polarity': 'Positive',
                'extracted_concern': 'financial management',
                'category': 'financial_stress',
                'intensity': 5,
                'keywords': ['budget', 'control', 'plan']
            },
            {
                'text': 'Constant worry about paying bills and rent',
                'polarity': 'Negative',
                'extracted_concern': 'financial worry',
                'category': 'financial_stress',
                'intensity': 8,
                'keywords': ['bills', 'rent', 'worry']
            },
            {
                'text': 'Job loss has depleted all my savings',
                'polarity': 'Negative',
                'extracted_concern': 'financial crisis',
                'category': 'financial_stress',
                'intensity': 9,
                'keywords': ['job loss', 'savings', 'depleted']
            },
            {
                'text': 'Started side hustle to manage financial pressure',
                'polarity': 'Neutral',
                'extracted_concern': 'financial coping',
                'category': 'financial_stress',
                'intensity': 6,
                'keywords': ['side hustle', 'financial', 'pressure']
            },

            # Work Stress
            {
                'text': 'Burnout from excessive workload and overtime',
                'polarity': 'Negative',
                'extracted_concern': 'work burnout',
                'category': 'work_stress',
                'intensity': 8,
                'keywords': ['burnout', 'workload', 'overtime']
            },
            {
                'text': 'Toxic workplace environment affecting mental health',
                'polarity': 'Negative',
                'extracted_concern': 'workplace environment',
                'category': 'work_stress',
                'intensity': 9,
                'keywords': ['toxic', 'workplace', 'mental health']
            },
            {
                'text': 'Set better boundaries at work, feeling relieved',
                'polarity': 'Positive',
                'extracted_concern': 'work boundaries',
                'category': 'work_stress',
                'intensity': 4,
                'keywords': ['boundaries', 'work', 'relieved']
            },
            {
                'text': 'Performance review anxiety affecting sleep',
                'polarity': 'Negative',
                'extracted_concern': 'work performance',
                'category': 'work_stress',
                'intensity': 7,
                'keywords': ['performance review', 'anxiety', 'sleep']
            },
            {
                'text': 'Difficulty maintaining work-life balance',
                'polarity': 'Negative',
                'extracted_concern': 'work-life balance',
                'category': 'work_stress',
                'intensity': 7,
                'keywords': ['work-life balance', 'difficulty']
            },

            # Family Issues
            {
                'text': 'Parents\' divorce causing emotional turmoil',
                'polarity': 'Negative',
                'extracted_concern': 'family conflict',
                'category': 'family_issues',
                'intensity': 8,
                'keywords': ['divorce', 'parents', 'emotional']
            },
            {
                'text': 'Sibling rivalry creating family tension',
                'polarity': 'Negative',
                'extracted_concern': 'family dynamics',
                'category': 'family_issues',
                'intensity': 6,
                'keywords': ['sibling', 'rivalry', 'tension']
            },
            {
                'text': 'Family therapy helping improve communication',
                'polarity': 'Positive',
                'extracted_concern': 'family improvement',
                'category': 'family_issues',
                'intensity': 5,
                'keywords': ['family therapy', 'communication', 'improve']
            },
            {
                'text': 'Caring for sick parent while managing work',
                'polarity': 'Negative',
                'extracted_concern': 'caregiver stress',
                'category': 'family_issues',
                'intensity': 8,
                'keywords': ['caring', 'sick parent', 'work']
            },
            {
                'text': 'Cultural differences causing family conflicts',
                'polarity': 'Negative',
                'extracted_concern': 'cultural issues',
                'category': 'family_issues',
                'intensity': 7,
                'keywords': ['cultural', 'differences', 'conflicts']
            },

            # Addiction
            {
                'text': 'Struggling to overcome gaming addiction',
                'polarity': 'Negative',
                'extracted_concern': 'behavioral addiction',
                'category': 'addiction',
                'intensity': 7,
                'keywords': ['gaming', 'addiction', 'struggling']
            },
            {
                'text': 'Three months sober but facing strong urges',
                'polarity': 'Neutral',
                'extracted_concern': 'recovery challenge',
                'category': 'addiction',
                'intensity': 6,
                'keywords': ['sober', 'urges', 'months']
            },
            {
                'text': 'Support group helping maintain sobriety',
                'polarity': 'Positive',
                'extracted_concern': 'recovery support',
                'category': 'addiction',
                'intensity': 5,
                'keywords': ['support group', 'sobriety', 'maintain']
            },
            {
                'text': 'Social media addiction affecting productivity',
                'polarity': 'Negative',
                'extracted_concern': 'digital addiction',
                'category': 'addiction',
                'intensity': 7,
                'keywords': ['social media', 'addiction', 'productivity']
            },
            {
                'text': 'Relapsed after six months, feeling ashamed',
                'polarity': 'Negative',
                'extracted_concern': 'relapse',
                'category': 'addiction',
                'intensity': 9,
                'keywords': ['relapsed', 'months', 'ashamed']
            },

            # PTSD/Trauma
            {
                'text': 'Flashbacks getting more intense lately',
                'polarity': 'Negative',
                'extracted_concern': 'trauma symptoms',
                'category': 'ptsd_trauma',
                'intensity': 9,
                'keywords': ['flashbacks', 'intense']
            },
            {
                'text': 'EMDR therapy helping process trauma',
                'polarity': 'Positive',
                'extracted_concern': 'trauma treatment',
                'category': 'ptsd_trauma',
                'intensity': 5,
                'keywords': ['EMDR', 'therapy', 'trauma']
            },
            {
                'text': 'Nightmares about past trauma affecting sleep',
                'polarity': 'Negative',
                'extracted_concern': 'trauma nightmares',
                'category': 'ptsd_trauma',
                'intensity': 8,
                'keywords': ['nightmares', 'trauma', 'sleep']
            },
            {
                'text': 'Triggered by loud noises and crowds',
                'polarity': 'Negative',
                'extracted_concern': 'trauma triggers',
                'category': 'ptsd_trauma',
                'intensity': 7,
                'keywords': ['triggered', 'loud noises', 'crowds']
            },
            {
                'text': 'Learning grounding techniques for flashbacks',
                'polarity': 'Neutral',
                'extracted_concern': 'coping strategies',
                'category': 'ptsd_trauma',
                'intensity': 6,
                'keywords': ['grounding', 'techniques', 'flashbacks']
            },

            # Identity Issues
            {
                'text': 'Questioning my identity and life purpose',
                'polarity': 'Negative',
                'extracted_concern': 'identity crisis',
                'category': 'identity_issues',
                'intensity': 7,
                'keywords': ['questioning', 'identity', 'purpose']
            },
            {
                'text': 'Struggling with cultural identity between two worlds',
                'polarity': 'Negative',
                'extracted_concern': 'cultural identity',
                'category': 'identity_issues',
                'intensity': 7,
                'keywords': ['cultural', 'identity', 'struggling']
            },
            {
                'text': 'Starting to accept and embrace who I am',
                'polarity': 'Positive',
                'extracted_concern': 'self-acceptance',
                'category': 'identity_issues',
                'intensity': 4,
                'keywords': ['accept', 'embrace', 'who I am']
            },
            {
                'text': 'Gender identity exploration causing family tension',
                'polarity': 'Negative',
                'extracted_concern': 'gender identity',
                'category': 'identity_issues',
                'intensity': 8,
                'keywords': ['gender', 'identity', 'family']
            },
            {
                'text': 'Finding community with similar identity struggles',
                'polarity': 'Positive',
                'extracted_concern': 'identity support',
                'category': 'identity_issues',
                'intensity': 5,
                'keywords': ['community', 'identity', 'similar']
            },

            # Cultural Adjustment
            {
                'text': 'Feeling isolated in new country, missing home',
                'polarity': 'Negative',
                'extracted_concern': 'cultural isolation',
                'category': 'cultural_adjustment',
                'intensity': 8,
                'keywords': ['isolated', 'new country', 'home']
            },
            {
                'text': 'Language barrier making it hard to connect',
                'polarity': 'Negative',
                'extracted_concern': 'communication barrier',
                'category': 'cultural_adjustment',
                'intensity': 7,
                'keywords': ['language barrier', 'connect']
            },
            {
                'text': 'Made first local friend, feeling more settled',
                'polarity': 'Positive',
                'extracted_concern': 'cultural integration',
                'category': 'cultural_adjustment',
                'intensity': 4,
                'keywords': ['local friend', 'settled']
            },
            {
                'text': 'Cultural differences at work causing stress',
                'polarity': 'Negative',
                'extracted_concern': 'workplace culture',
                'category': 'cultural_adjustment',
                'intensity': 6,
                'keywords': ['cultural differences', 'work', 'stress']
            },
            {
                'text': 'Starting to appreciate new cultural experiences',
                'polarity': 'Positive',
                'extracted_concern': 'cultural adaptation',
                'category': 'cultural_adjustment',
                'intensity': 5,
                'keywords': ['appreciate', 'cultural', 'experiences']
            },

            # Grief/Loss
            {
                'text': 'Can\'t cope with the loss of my parent',
                'polarity': 'Negative',
                'extracted_concern': 'grief processing',
                'category': 'grief_loss',
                'intensity': 9,
                'keywords': ['loss', 'parent', 'cope']
            },
            {
                'text': 'First holiday season without them is unbearable',
                'polarity': 'Negative',
                'extracted_concern': 'holiday grief',
                'category': 'grief_loss',
                'intensity': 9,
                'keywords': ['holiday', 'without', 'unbearable']
            },
            {
                'text': 'Grief support group helping me process loss',
                'polarity': 'Neutral',
                'extracted_concern': 'grief support',
                'category': 'grief_loss',
                'intensity': 6,
                'keywords': ['grief', 'support group', 'process']
            },
            {
                'text': 'Finding ways to honor their memory',
                'polarity': 'Positive',
                'extracted_concern': 'memorial coping',
                'category': 'grief_loss',
                'intensity': 5,
                'keywords': ['honor', 'memory', 'ways']
            },
            {
                'text': 'Unexpected triggers bringing back grief waves',
                'polarity': 'Negative',
                'extracted_concern': 'grief triggers',
                'category': 'grief_loss',
                'intensity': 8,
                'keywords': ['triggers', 'grief', 'waves']
            },

            # Burnout
            {
                'text': 'Complete physical and emotional exhaustion',
                'polarity': 'Negative',
                'extracted_concern': 'severe burnout',
                'category': 'burnout',
                'intensity': 9,
                'keywords': ['exhaustion', 'physical', 'emotional']
            },
            {
                'text': 'Taking a sabbatical to recover from burnout',
                'polarity': 'Neutral',
                'extracted_concern': 'burnout recovery',
                'category': 'burnout',
                'intensity': 6,
                'keywords': ['sabbatical', 'recover', 'burnout']
            },
            {
                'text': 'Started setting boundaries to prevent burnout',
                'polarity': 'Positive',
                'extracted_concern': 'burnout prevention',
                'category': 'burnout',
                'intensity': 5,
                'keywords': ['boundaries', 'prevent', 'burnout']
            },
            {
                'text': 'Can\'t focus or be productive anymore',
                'polarity': 'Negative',
                'extracted_concern': 'cognitive burnout',
                'category': 'burnout',
                'intensity': 8,
                'keywords': ['focus', 'productive', 'anymore']
            },
            {
                'text': 'Feeling disconnected from work and colleagues',
                'polarity': 'Negative',
                'extracted_concern': 'work disconnection',
                'category': 'burnout',
                'intensity': 7,
                'keywords': ['disconnected', 'work', 'colleagues']
            },

            # Positive Outlook
            {
                'text': 'Therapy and medication finally showing results',
                'polarity': 'Positive',
                'extracted_concern': 'treatment success',
                'category': 'positive_outlook',
                'intensity': 4,
                'keywords': ['therapy', 'medication', 'results']
            },
            {
                'text': 'Learning to appreciate small daily victories',
                'polarity': 'Positive',
                'extracted_concern': 'gratitude practice',
                'category': 'positive_outlook',
                'intensity': 3,
                'keywords': ['appreciate', 'victories', 'daily']
            },
            {
                'text': 'Started journaling and seeing patterns improve',
                'polarity': 'Positive',
                'extracted_concern': 'self-improvement',
                'category': 'positive_outlook',
                'intensity': 4,
                'keywords': ['journaling', 'patterns', 'improve']
            },
            {
                'text': 'Finding hope and strength in support system',
                'polarity': 'Positive',
                'extracted_concern': 'support appreciation',
                'category': 'positive_outlook',
                'intensity': 3,
                'keywords': ['hope', 'strength', 'support']
            },
            {
                'text': 'Meditation practice helping maintain balance',
                'polarity': 'Positive',
                'extracted_concern': 'wellness practice',
                'category': 'positive_outlook',
                'intensity': 3,
                'keywords': ['meditation', 'balance', 'practice']
            }
        ]

    def train_system(self):
        self.analyzer.train(self.training_data)

# FastAPI endpoints


@app.post("/users/", response_model=dict)
async def create_user(user: User):
    """Create a new user in the system"""
    try:
        db_manager.add_user(user.user_id, user.name)
        return {"message": "User created successfully", "user_id": user.user_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/chat/{user_id}", response_model=dict)
async def chat(user_id: str, message: Message):
    """Process a chat message and return analysis with model response"""
    try:
        system = MentalHealthAnalysisSystem()
        analysis = system.analyzer.analyze_text(
            message.text,
            datetime.now()
        )
        model_response = await model_manager.get_response(user_id, message.text)
        db_manager.add_conversation(
            user_id, message.text, analysis, model_response)
        return {
            "message": message.text,
            "analysis": analysis,
            "model_response": model_response
        }
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while processing your request") from e


@app.get("/history/{user_id}", response_model=ConversationHistory)
async def get_history(user_id: str):
    """Get conversation history for a user"""
    try:
        history = db_manager.get_conversation_history(user_id)
        return ConversationHistory(messages=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close_conversation/{user_id}")
async def close_conversation(user_id: str):
    """Close a user's conversation session"""
    try:
        model_manager.close_conversation(user_id)
        return {"message": "Conversation closed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analyze_trends/{user_id}")
async def analyze_trends(user_id: str):
    """Analyze trends in user's conversation history"""
    try:
        history = db_manager.get_conversation_history(user_id)
        if not history:
            return {"message": "No conversation history found"}

        # Convert history to DataFrame
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Calculate trends
        sentiment_trend = df.apply(
            lambda x: json.loads(x['analysis'])[
                'sentiment']['scores']['compound']
            if isinstance(x['analysis'], str)
            else x['analysis']['sentiment']['scores']['compound'],
            axis=1
        ).mean()

        categories = df.apply(
            lambda x: json.loads(x['analysis'])['category']['primary']
            if isinstance(x['analysis'], str)
            else x['analysis']['category']['primary'],
            axis=1
        ).value_counts().to_dict()

        return {
            "total_messages": len(history),
            "average_sentiment": float(sentiment_trend),
            "category_distribution": categories,
            "time_span": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Shutdown event handler


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    # Close all active conversations
    for user_id in list(model_manager.conversations.keys()):
        model_manager.close_conversation(user_id)


# Utility functions for analysis and reporting
def generate_user_report(user_id: str) -> dict:
    """Generate a comprehensive report for a user"""
    history = db_manager.get_conversation_history(user_id)
    if not history:
        return {"message": "No conversation history found"}

    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Analyze sentiment progression
    df['sentiment'] = df.apply(
        lambda x: json.loads(x['analysis'])['sentiment']['scores']['compound']
        if isinstance(x['analysis'], str)
        else x['analysis']['sentiment']['scores']['compound'],
        axis=1
    )

    # Analyze categories
    df['category'] = df.apply(
        lambda x: json.loads(x['analysis'])['category']['primary']
        if isinstance(x['analysis'], str)
        else x['analysis']['category']['primary'],
        axis=1
    )

    # Calculate weekly averages
    weekly_stats = df.set_index('timestamp').resample('W').agg({
        'sentiment': 'mean',
        'category': lambda x: x.mode()[0] if not x.empty else None
    })

    return {
        "overview": {
            "total_conversations": len(history),
            "time_period": {
                "start": df['timestamp'].min().isoformat(),
                "end": df['timestamp'].max().isoformat()
            },
            "average_sentiment": float(df['sentiment'].mean())
        },
        "weekly_progression": weekly_stats.to_dict(orient='index'),
        "category_distribution": df['category'].value_counts().to_dict(),
        "sentiment_progression": {
            "trend": "Improving" if df['sentiment'].iloc[-5:].mean() > df['sentiment'].iloc[:5].mean() else "Declining",
            "variance": float(df['sentiment'].var())
        }
    }

# Main execution


def main():
    # Configuration
    host = "0.0.0.0"
    port = 8000

    # Print startup message
    print(f"""
    ===============================================
    Mental Health Analysis System
    ===============================================

    Server starting up...

    Available endpoints:
    - POST   /users/                  Create new user
    - POST   /chat/{{user_id}}           Send message
    - GET    /history/{{user_id}}        Get chat history
    - POST   /close_conversation/{{user_id}} Close session
    - GET    /analyze_trends/{{user_id}} Get analysis

    Server will be available at:
    http://{host}:{port}

    Documentation will be available at:
    http://{host}:{port}/docs

    Press Ctrl+C to stop the server
    ===============================================
    """)

    # Run server
    uvicorn.run(app, host=host, port=port)

# Example usage


def example_usage():
    """Example of how to use the system"""
    async def run_example():
        # Create a user
        user_id = "user123"
        user = User(user_id=user_id, name="Test User")
        await create_user(user)

        # Send some messages
        messages = [
            "I've been feeling anxious about work lately",
            "Started meditation today, feeling a bit better",
            "Having trouble sleeping due to stress",
        ]

        for message_text in messages:
            message = Message(text=message_text)
            response = await chat(user_user_id, message)
            print(f"\nMessage: {message_text}")
            print(f"Analysis: {json.dumps(response['analysis'], indent=2)}")
            print(f"Model Response: {response['model_response']}")

        # Get history
        history = await get_history(user_user_id)
        print("\nConversation History:")
        print(json.dumps(history.dict(), indent=2))

        # Get trends
        trends = await analyze_trends(user_user_id)
        print("\nTrends Analysis:")
        print(json.dumps(trends, indent=2))

        # Close conversation
        await close_conversation(user_user_id)

    # Run the example
    asyncio.run(run_example())


if __name__ == "__main__":
    # Choose whether to run the server or example usage
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_usage()
    else:
        main()

"""
To use this system:

1. Install required packages:
   pip install fastapi uvicorn sqlite3 torch spacy vaderSentiment \
               scikit-learn pandas numpy matplotlib seaborn plotly

2. Download spaCy model:
   python -m spacy download en_core_web_sm

3. Install Ollama and download the llama2 model:
   ollama pull llama2

4. Run the server:
   python script.py

   Or run the example:
   python script.py --example

5. Access the API documentation:
   http://localhost:8000/docs

Example curl commands:

# Create a user
curl -X POST "http://localhost:8000/users/" \
     -H "Content-Type: application/json" \
     -d '{"user_id": "user123", "name": "John Doe"}'

# Send a message
curl -X POST "http://localhost:8000/chat/user123" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am feeling anxious today"}'

# Get conversation history
curl "http://localhost:8000/history/user123"

# Get trends analysis
curl "http://localhost:8000/analyze_trends/user123"

# Close conversation
curl - X POST "http://localhost:8000/close_conversation/user123"
"""
