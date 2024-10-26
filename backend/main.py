# Install required packages
# !pip install transformers torch spacy pandas numpy scikit-learn vaderSentiment matplotlib seaborn plotly
# !python - m spacy download en_core_web_sm

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import spacy
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import json
training_data = [
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

timeline_entries = [{'text': 'I feel completely overwhelmed and anxious',
                     'timestamp': '2024-01-01'},
                    {'text': 'Started therapy, feeling a bit better',
                     'timestamp': '2024-01-15'},
                    {'text': 'Making good progress, still some anxiety',
                     'timestamp': '2024-01-30'}]


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
        """Get a list of all available mental health categories"""
        return list(cls.CATEGORIES.keys())

    @classmethod
    def get_keywords(cls, category):
        """Get keywords associated with a specific mental health category"""
        return cls.CATEGORIES.get(category, [])


class MentalHealthAnalyzer:
    """Class to perform mental health analysis on text data"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.intensity_scaler = MinMaxScaler()
        self.timeline_data = []
        self.categories = MentalHealthCategories.get_all_categories()

    def train(self, training_data):
        """Train the mental health analyzer with labeled training data"""
        texts = [item['text'] for item in training_data]
        categories = [item['category'] for item in training_data]
        intensities = [item['intensity'] for item in training_data]

        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, categories)
        self.intensity_scaler.fit(np.array(intensities).reshape(-1, 1))

    def analyze_text(self, text, timestamp=None):
        """Analyze a single text entry for mental health insights"""
        text = text.strip()
        timestamp = timestamp or datetime.now()

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
            'timestamp': timestamp,
            'sentiment': {
                'label': sentiment,
                'scores': sentiment_scores
            },
            'category': {
                'primary': category,
                'confidence': float(max(probs)),
                'all_probabilities': dict(zip(self.classifier.classes_, probs))
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
        """Extract relevant keywords from text using NER and predefined terms"""
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
        """Calculate intensity of mental health concerns based on text and sentiment"""
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

    def analyze_timeline(self):
        """Analyze the entire timeline data for trends and insights"""
        if not self.timeline_data:
            return None

        timeline = pd.DataFrame(self.timeline_data)
        timeline['timestamp'] = pd.to_datetime(timeline['timestamp'])

        sentiment_trends = timeline.groupby('timestamp').apply(
            lambda x: pd.Series({
                'avg_sentiment': np.mean([d['scores']['compound'] for d in x['sentiment']]),
                'avg_intensity': np.mean([d['score'] for d in x['intensity']])
            })
        )

        return {
            'sentiment_trends': sentiment_trends,
            'total_entries': len(timeline),
            'date_range': {
                'start': min(timeline['timestamp']),
                'end': max(timeline['timestamp'])
            },
            'category_distribution': timeline['category'].apply(lambda x: x['primary']).value_counts().to_dict(),
            'average_intensity': np.mean([d['intensity']['score'] for d in self.timeline_data])
        }


class MentalHealthVisualizer:
    """Class to generate visualizations for mental health analysis"""

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
        """Plot distribution of mental health categories"""
        categories = [entry['category']['primary']
                      for entry in self.analyzer.timeline_data]
        category_counts = pd.Series(categories).value_counts()

        plt.figure(figsize=(12, 6))
        sns.barplot(x=category_counts.index, y=category_counts.values)
        plt.title('Distribution of Mental Health Categories')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_intensity_heatmap(self):
        """Plot heatmap of intensity scores by category and date"""
        timeline_data = pd.DataFrame(self.analyzer.timeline_data)
        timeline_data['intensity_score'] = timeline_data['intensity'].apply(
            lambda x: x['score']
        )
        timeline_data['category'] = timeline_data['category'].apply(
            lambda x: x['primary']
        )

        pivot_table = pd.pivot_table(
            timeline_data,
            values='intensity_score',
            index='category',
            columns=pd.to_datetime(timeline_data['timestamp']).dt.date,
            aggfunc='mean'
        )

        plt.figure(figsize=(15, 8))
        sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.1f')
        plt.title('Intensity Heatmap by Category and Date')
        plt.tight_layout()
        plt.show()

    def generate_interactive_dashboard(self):
        """Generate an interactive dashboard for mental health analysis"""
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


class DataExporter:
    """Class to export mental health analysis data to CSV, JSON, etc."""

    def __init__(self, analyzer):
        self.analyzer = analyzer

    def export_to_csv(self, filename='mental_health_analysis.csv'):
        df = pd.DataFrame(self.analyzer.timeline_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['sentiment_label'] = df['sentiment'].apply(lambda x: x['label'])
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: x['scores']['compound'])
        df['category'] = df['category'].apply(lambda x: x['primary'])
        df['intensity_score'] = df['intensity'].apply(lambda x: x['score'])
        df['keywords'] = df['keywords'].apply(
            lambda x: ', '.join([k['text'] for k in x]))

        columns = ['timestamp', 'text', 'sentiment_label', 'sentiment_score',
                   'category', 'intensity_score', 'keywords']
        df[columns].to_csv(filename, index=False)
        print(f"Data exported to {filename}")

    def export_to_json(self, filename='mental_health_analysis.json'):
        """Export mental health analysis data to JSON format"""
        with open(filename, 'w') as f:
            json.dump(self.analyzer.timeline_data, f, default=str, indent=2)
        print(f"Data exported to {filename}")

    def generate_summary_report(self, filename='analysis_summary.txt'):
        """Generate a summary report of the mental health analysis"""
        timeline_analysis = self.analyzer.analyze_timeline()

        with open(filename, 'w') as f:
            f.write("Mental Health Analysis Summary Report\n")
            f.write("=" * 40 + "\n\n")

            f.write("Overview:\n")
            f.write(f"Total Entries: {timeline_analysis['total_entries']}\n")
            f.write(f"Date Range: {timeline_analysis['date_range']['start']} to {
                    timeline_analysis['date_range']['end']}\n")
            f.write(f"Average Intensity: {
                    timeline_analysis['average_intensity']:.2f}\n\n")

            f.write("Category Distribution:\n")
            for category, count in timeline_analysis['category_distribution'].items():
                f.write(f"{category}: {count} entries\n")

        print(f"Summary report generated: {filename}")


def create_sample_timeline_data(num_days=30):
    """Create sample timeline data for testing"""
    base_date = datetime.now() - timedelta(days=num_days)
    sample_texts = [
        "Feeling very anxious about everything today",
        "Had a good therapy session, feeling hopeful",
        "Struggling with sleep again",
        "Making progress with my anxiety management",
        "Feeling down and unmotivated",
        "Started a new exercise routine, feeling better",
        "Having trouble concentrating at work",
        "Family issues causing stress",
        "Feeling more confident after counseling",
        "Worried about my health constantly"
    ]

    timeline_entries = []
    for i in range(num_days):
        current_date = base_date + timedelta(days=i)
        text = np.random.choice(sample_texts)
        timeline_entries.append({
            'text': text,
            'timestamp': current_date
        })

    return timeline_entries


class MentalHealthAnalysisSystem:
    """Class to manage the end-to-end mental health analysis system"""

    def __init__(self):
        self.analyzer = MentalHealthAnalyzer()
        self.visualizer = MentalHealthVisualizer(self.analyzer)
        self.exporter = DataExporter(self.analyzer)

    def train_system(self, training_data):
        print("Training the system...")
        self.analyzer.train(training_data)
        print("Training completed!")

    def process_timeline(self, timeline_entries):
        print("Processing timeline entries...")
        for entry in timeline_entries:
            self.analyzer.analyze_text(entry['text'],
                                       pd.to_datetime(entry['timestamp']))
        print("Timeline processing completed!")

    def generate_comprehensive_report(self, output_dir='analysis_output'):
        """Generate comprehensive analysis with visualizations and exports"""
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Export data
        self.exporter.export_to_csv(f"{output_dir}/analysis_data.csv")
        self.exporter.export_to_json(f"{output_dir}/analysis_data.json")
        self.exporter.generate_summary_report(
            f"{output_dir}/summary_report.txt")

        # Generate visualizations
        print("\nGenerating visualizations...")

        # Save static plots
        plt.figure()
        self.visualizer.plot_sentiment_timeline()
        plt.savefig(f"{output_dir}/sentiment_timeline.png")

        plt.figure()
        self.visualizer.plot_category_distribution()
        plt.savefig(f"{output_dir}/category_distribution.png")

        plt.figure()
        self.visualizer.plot_intensity_heatmap()
        plt.savefig(f"{output_dir}/intensity_heatmap.png")

        # Generate interactive dashboard
        self.visualizer.generate_interactive_dashboard()

        print(f"Analysis outputs saved to {output_dir}/")


def main():
    # Initialize the system
    system = MentalHealthAnalysisSystem()

    # Train the system with provided training data
    system.train_system(training_data)

    # Process timeline data
    # Option 1: Use provided timeline entries
    # system.process_timeline(timeline_entries)

    # Option 2: Generate sample timeline data
    sample_timeline = create_sample_timeline_data(30)
    system.process_timeline(sample_timeline)

    # Generate comprehensive report
    system.generate_comprehensive_report()

    # Example of real-time analysis
    print("\nReal-time Analysis Example:")
    result = system.analyzer.analyze_text(
        "i have exams next week and im so tired i am unable to understand anything im so scared",
        datetime.now()
    )

    print("\nAnalysis Result:")
    print(f"Sentiment: {result['sentiment']['label']}")
    print(f"Category: {result['category']['primary']}")
    print(f"Intensity: {result['intensity']['level']
                        } ({result['intensity']['score']})")
    print(f"Keywords: {[k['text'] for k in result['keywords']]}")

    print("---")

    print(result)

    print("---")

    print(sorted(result['category']['all_probabilities'].items(
    ), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    main()

# Additional utility functions for specific analyses


def analyze_category_progression(analyzer, category):
    """Analyze progression of a specific mental health category"""
    category_entries = [
        entry for entry in analyzer.timeline_data
        if entry['category']['primary'] == category
    ]

    if not category_entries:
        return None

    return {
        'total_entries': len(category_entries),
        'average_intensity': np.mean([e['intensity']['score'] for e in category_entries]),
        'sentiment_progression': [e['sentiment']['scores']['compound'] for e in category_entries],
        'timestamps': [e['timestamp'] for e in category_entries]
    }


def generate_weekly_summary(analyzer):
    """Generate weekly summary of mental health patterns"""
    df = pd.DataFrame(analyzer.timeline_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['week'] = df['timestamp'].dt.isocalendar().week

    weekly_stats = df.groupby('week').agg({
        'sentiment': lambda x: np.mean([i['scores']['compound'] for i in x]),
        'intensity': lambda x: np.mean([i['score'] for i in x]),
        'category': lambda x: pd.Series([i['primary'] for i in x]).mode()[0]
    }).reset_index()

    return weekly_stats


def export_to_excel(analyzer, filename='mental_health_analysis.xlsx'):
    """Export detailed analysis to Excel with multiple sheets"""
    with pd.ExcelWriter(filename) as writer:
        # Daily entries
        daily_df = pd.DataFrame(analyzer.timeline_data)
        daily_df['timestamp'] = pd.to_datetime(daily_df['timestamp'])
        daily_df.to_excel(writer, sheet_name='Daily_Entries', index=False)

        # Weekly summary
        weekly_summary = generate_weekly_summary(analyzer)
        weekly_summary.to_excel(
            writer, sheet_name='Weekly_Summary', index=False)

        # Category analysis
        category_stats = pd.DataFrame([
            analyze_category_progression(analyzer, cat)
            for cat in MentalHealthCategories.get_all_categories()
        ])
        category_stats.to_excel(
            writer, sheet_name='Category_Analysis', index=False)

# Example usage of additional analyses


def run_advanced_analysis():
    """Run advanced analysis on the mental health data"""

    system = MentalHealthAnalysisSystem()
    system.train_system(training_data)
    system.process_timeline(create_sample_timeline_data())

    # Generate weekly summary
    weekly_summary = generate_weekly_summary(system.analyzer)
    print("\nWeekly Summary:")
    print(weekly_summary)

    # Analyze specific category
    anxiety_progression = analyze_category_progression(
        system.analyzer, 'anxiety')
    print("\nAnxiety Progression Analysis:")
    print(anxiety_progression)

    # Export detailed Excel report
    export_to_excel(system.analyzer)

# Run advanced analysis if needed
# run_advanced_analysis()
