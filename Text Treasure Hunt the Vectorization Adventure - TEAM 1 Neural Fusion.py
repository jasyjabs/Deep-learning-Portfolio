#!/usr/bin/env python
# coding: utf-8

# ### Team1 - Neural Fusion
# 
# 
# ### Introduction 
# 
# Objective:
# 
# To solve a series of challenges, each leading closer to uncovering the final treasure. 
# 
# Along the way, we are to employ text processing techniques to decode hidden clues embedded within texts,to decode clues, uncover hidden connections, and collaborate with others to reach the ultimate treasure! 
# 
# Clues for Group 1
# 1. Clue 1a (Easier): From lines and shapes to scenes so bright, I give digital creations the illusion of light.
# 2. Clue 2a (Harder): Where polygons dance and shaders weave, my artistry helps the virtual world breathe.
# 3. Clue 1 b(Easier): Tiny squares, a canvas wide, building images side by side.
# 4. Clue 2b (Harder): RGB spells my secret code, the building blocks where colors explode.
# 
# 

# ### Install the necessary libraries - Setting Up 
# 
# #ensuring we have all the necessary libraries installed and importing them into our Python environment
# 
# #The libraries include : NLTK, Pandas, Scikit-learn, Gensim, and Spacy. Additionally, we provide an optional section for advanced exploration with Transformers, which requires installation and importation of the Transformers library.
# 
# 

# In[1]:



get_ipython().system('pip install torch')
get_ipython().system('pip install --upgrade torch transformers')
get_ipython().system('pip install Matplotlib')
get_ipython().system('pip install seaborn')


# In[2]:


# Import libraries
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import nltk

import numpy as np
import pandas as pd
import math
import os
#import system

import matplotlib.pyplot as plt #data viz
import seaborn as sns # data viz
#supress warnings
import warnings
warnings.filterwarnings('ignore')

nltk.download("stopwords")
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec  # For Word2Vec embeddings
import re  # For regular expressions

# Optional advanced exploration with Transformers
from transformers import pipeline


# # Quest Begins – The Initial Clue
# *Clue 1a (Easier): From lines and shapes to scenes so bright, I give digital creations the illusion of light.

# In[3]:


# Load the 20 newsgroups dataset for 'sci.med' and 'sci.space' categories

categories = ["sci.med", "sci.space"]
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)

# Create a dataframe for 'sci.med' and 'sci.space' categories
df = pd.DataFrame(data=newsgroups_train.data, columns=["text"])
df["target"] = newsgroups_train.target
df["category"] = df["target"].map(lambda x: categories[x])

# Display the number of rows and columns
df.shape


# In[4]:


# Display the first few rows to test
df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.describe().T


# # Keyword Quest
# # Starting with TF-IDF:
# 

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity
from nltk.corpus import stopwords

def extract_keywords(text_data, clues, top_n=10):
    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    preprocessed_data = [' '.join([word for word in text.lower().split() if word not in stop_words]) for text in text_data]

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_data)

    # Get the feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()

    # Preprocess the clues
    preprocessed_clues = [' '.join([word for word in clue.lower().split() if word not in stop_words]) for clue in clues]

    # Transform the clues using the same vectorizer
    clue_vectors = vectorizer.transform(preprocessed_clues)

    # Calculate cosine similarity between clues and text data
    similarity_scores = cosine_similarity(clue_vectors, tfidf_matrix)

    # Find the top N most similar keywords for each clue
    top_keywords = []
    for i in range(len(clues)):
        clue_scores = similarity_scores[i]
        top_indices = clue_scores.argsort()[-top_n:][::-1]
        top_keywords.append([feature_names[index] for index in top_indices])

    return top_keywords

clues = [
    "From lines and shapes to scenes so bright, I give digital creations the illusion of light.",
    "Where polygons dance and shaders weave, my artistry helps the virtual world breathe."
]

text_data = [
    "Digital art encompasses a range of artistic work and practices that use digital technology as an essential part of the creative and/or presentation process.",
    "It involves creating artwork digitally, either by computer software or other digital tools.",
    "The resulting artwork can be printed, displayed on a screen, or projected onto a surface."
]

# Extract relevant keywords based on the clues
relevant_keywords = extract_keywords(text_data, clues)

# Print the relevant keywords for each clue
for i, clue in enumerate(clues):
    print(f"Clue: {clue}")
    print("Relevant keywords:", relevant_keywords[i])
    print()


# # Clues

# In[10]:


import re

# Clues
scimed_Clue_1a = (
    "From lines and shapes to scenes so bright, I give digital creations the illusion of light."
)
scimed_Clue_2a = (
    "Where polygons dance and shaders weave, my artistry helps the virtual world breathe."
)
scimed_Clue_1b = (
    "Tiny squares, a canvas wide, building images side by side."
)
scimed_Clue_2b = (
    "RGB spells my secret code, the building blocks where colors explode."
)


# Sci Space topic
scispace_Clue_1a = "From lines and shapes to scenes so bright, I give digital creations the illusion of light."
scispace_Clue_2a = "Where polygons dance and shaders weave, my artistry helps the virtual world breathe."
scispace_Clue_1b = (
    "Tiny squares, a canvas wide, building images side by side."
)
scispace_Clue_2b = "RGB spells my secret code, the building blocks where colors explode."

scimed_clues_list = [
    scimed_Clue_1a,
    scimed_Clue_2a,
    scimed_Clue_1b,
    scimed_Clue_2b,
    
]
scispace_clues_list = [
    scispace_Clue_1a,
    scispace_Clue_2a,
    scispace_Clue_1b,
    scispace_Clue_2b,
]

# clean the clues list text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

scimed_clues_list = [clean_text(clue) for clue in scimed_clues_list]
scispace_clues_list = [clean_text(clue) for clue in scispace_clues_list]

print(f"Sci Med Clues: {scimed_clues_list}")
print("----------------")
print(f"Sci Space Clues: {scispace_clues_list}")


# In[ ]:





# ### Word embeddings like Word2Vec or GloVe help us understand how words relate to each other.
# 
# # Topkey & Related  words

# In[11]:


from sklearn.feature_extraction.text
import TfidfVectorizer
import pandas as pd
import re

# Clues
scimed_Clue_1a = (
    "From lines and shapes to scenes so bright, I give digital creations the illusion of light."
)
scimed_Clue_2a = (
    "Where polygons dance and shaders weave, my artistry helps the virtual world breathe."
)
scimed_Clue_1b = (
    "Tiny squares, a canvas wide, building images side by side."
)
scimed_Clue_2b = (
    "RGB spells my secret code, the building blocks where colors explode."
)


# Sci Space topic
scispace_Clue_1a = "From lines and shapes to scenes so bright, I give digital creations the illusion of light."
scispace_Clue_2a = "Where polygons dance and shaders weave, my artistry helps the virtual world breathe."
scispace_Clue_1b = (
    "Tiny squares, a canvas wide, building images side by side."
)
scispace_Clue_2b = "RGB spells my secret code, the building blocks where colors explode."

scimed_clues_list = [
    scimed_Clue_1a,
    scimed_Clue_2a,
    scimed_Clue_1b,
    scimed_Clue_2b,
    
]
scispace_clues_list = [
    scispace_Clue_1a,
    scispace_Clue_2a,
    scispace_Clue_1b,
    scispace_Clue_2b,
]

# clean the clues list text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

scimed_clues_list = [clean_text(clue) for clue in scimed_clues_list]
scispace_clues_list = [clean_text(clue) for clue in scispace_clues_list]

def tfidf_extract_keywords(clues_list):
    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    # Fit the vectorizer to the clues
    tfidf_matrix = tfidf_vectorizer.fit_transform(clues_list)

    # Get the feature names of `tfidf_vectorizer`
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a DataFrame of the `tfidf_matrix`
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Get the top feature from each clue
    top_keywords = tfidf_df.idxmax(axis=1).values
    return top_keywords

scimed_related_keywords_tfidf = tfidf_extract_keywords(scimed_clues_list)
scispace_related_keywords_tfidf = tfidf_extract_keywords(scispace_clues_list)

print(f"Top keywords for 'sci.med' clues: {scimed_related_keywords_tfidf}")
print("----------------")
print(f"Top keywords for 'sci.space' clues: {scispace_related_keywords_tfidf}")


# # Glove embeddings 

# In[15]:


from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Load pre-trained GloVe embeddings
glove_file = "20newsgroups.txt"
glove_embeddings = KeyedVectors.load_word2vec_format(
    glove_file, binary=False, no_header=True
)


def tfidf_extract_keywords(clues_list):
    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(stop_words="english")

    # Fit the vectorizer to the clues
    tfidf_matrix = tfidf_vectorizer.fit_transform(clues_list)

    # Get the feature names of `tfidf_vectorizer`
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Create a DataFrame of the `tfidf_matrix`
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Get the top feature from each clue
    top_keywords = tfidf_df.idxmax(axis=1).values
    return top_keywords

# Make sure to define scimed_clues_list and scispace_clues_list with correct clues before calling tfidf_extract_keywords function
scimed_related_keywords_glove = tfidf_extract_keywords(scimed_clues_list)
scispace_related_keywords_glove = tfidf_extract_keywords(scispace_clues_list)

print(f"Related keywords for 'sci.med' clues (GloVe): {scimed_related_keywords_glove}")
print("----------------")
print(f"Related keywords for 'sci.space' clues (GloVe): {scispace_related_keywords_glove}")


# ## reveal new concepts, or hint at hidden patterns within the data.
# ## Word2Vec 

# In[19]:


from gensim.models import Word2Vec

# Function to get related keywords using Word2Vec
def get_related_keywords(clues_list, keywords, n=5):
    # Tokenize the clues
    tokenized_clues = [clue.split() for clue in clues_list]

    # Create a Word2Vec model
    word2vec = Word2Vec(tokenized_clues, vector_size=100, window=5, min_count=1, sg=1)

    related_keywords = []
    for keyword in keywords:
        try:
            # Get the most similar keywords for each input keyword
            related = word2vec.wv.most_similar(positive=[keyword], topn=n)
            related_keywords.extend([w[0] for w in related])
        except KeyError:
            # Skip keywords not in the vocabulary
            pass

    return related_keywords


# Define scimed_keywords (example)
scimed_keywords = ['artistry', 'blocks', 'breathe', 'bright', 'building']

# Get related keywords for 'sci.med' clues
scimed_related_keywords = get_related_keywords(scimed_clues_list, scimed_keywords)

print(f"Related keywords for 'sci.med' clues: {scimed_related_keywords}")


# # Semantic Safari
# 
# ## Exploring the World of Meaning: Word embeddings like Word2Vec or GloVe help us understand how words relate to each other.
# 

# In[21]:


from gensim.models import Word2Vec

# Function to calculate similarity between keywords and clues
def calculate_similarity(keywords, clues_list):
    # Tokenize the clues
    tokenized_clues = [clue.split() for clue in clues_list]

    # Create a Word2Vec model
    word2vec = Word2Vec(tokenized_clues, vector_size=100, window=5, min_count=1, sg=1)

    similarities = []
    for keyword in keywords:
        try:
            # Get the most similar clues for each keyword
            similar_clues = word2vec.wv.most_similar(positive=[keyword], topn=1)
            similarities.append((keyword, similar_clues[0][0], similar_clues[0][1]))
        except KeyError:
            # Skip keywords not in the vocabulary
            pass

    return similarities

# Define scispace_keywords (example)
scispace_keywords = ['space', 'galaxy', 'astronomy', 'cosmos', 'planet']

# Calculate similarities between 'sci.med' keywords and 'sci.space' clues, and vice versa
scimed_similarities = calculate_similarity(scimed_keywords, scispace_clues_list)
scispace_similarities = calculate_similarity(scispace_keywords, scimed_clues_list)

print(
    f"Similarities between 'sci.med' keywords and 'sci.space' clues: {scimed_similarities}"
)
print("----------------")
print(
    f"Similarities between 'sci.space' keywords and 'sci.med' clues: {scispace_similarities}"
)


# 
# 
# ## Trying to Calculate similarities between our keywords and texts in other categories. 
# 

# In[22]:


# Calculate the average similarity score for 'sci.med' and 'sci.space' clues
scimed_similarity_score = np.mean([score for _, _, score in scimed_similarities])
scispace_similarity_score = np.mean([score for _, _, score in scispace_similarities])

print(f"Average similarity score for 'sci.med' clues: {scimed_similarity_score}")
print(f"Average similarity score for 'sci.space' clues: {scispace_similarity_score}")


# # why is the confidence score low? ***

# # Advanced Exploration: Transformers ** (Optional)
# 
# Transformer-based models provide even more nuanced semantic understanding. 

# In[23]:


from transformers import pipeline


# Function to perform question answering using a pre-trained model
def answer_question(question, context):
    answerer = pipeline("question-answering")
    answer = answerer({"question": question, "context": context})
    return answer


# Function to perform text classification using a pre-trained model
def classify_text(text):
    classifier = pipeline("text-classification")
    result = classifier(text)
    return result


# Example usage for question answering
scimed_question = "What influences mood in the brain?"
scimed_context = " ".join(scimed_clues_list)
scimed_answer = answer_question(scimed_question, scimed_context)
print(f"Question: {scimed_question}")
print(f"Answer: {scimed_answer['answer']}")
print(f"Score: {scimed_answer['score']}")
print("----------------")

scispace_question = "What is the goal of space exploration?"
scispace_context = " ".join(scispace_clues_list)
scispace_answer = answer_question(scispace_question, scispace_context)
print(f"Question: {scispace_question}")
print(f"Answer: {scispace_answer['answer']}")
print(f"Score: {scispace_answer['score']}")
print("----------------")

# Example usage for text classification
scimed_text = "This text seems to be about neurotransmitters and their effects on mood."
scimed_classification = classify_text(scimed_text)
print(f"Text: {scimed_text}")
print(f"Classification: {scimed_classification[0]['label']}")
print(f"Score: {scimed_classification[0]['score']}")
print("----------------")

scispace_text = (
    "This text seems to be about space exploration and technological advancements."
)
scispace_classification = classify_text(scispace_text)
print(f"Text: {scispace_text}")
print(f"Classification: {scispace_classification[0]['label']}")
print(f"Score: {scispace_classification[0]['score']}")


#  # Trying to identify the Pattern 
# 

# In[24]:


import re


##  Function to detect patterns in clues using regular expressions
def detect_patterns(clues_list):
    patterns = []
    for clue in clues_list:
        # Pattern for chemical names (e.g., serotonin, tryptophan)
        chemical_pattern = r"\b[A-Za-z]+\b"
        chemicals = re.findall(chemical_pattern, clue)

        # Pattern for scientific terms (e.g., neurotransmitter, microbiome)
        science_term_pattern = r"\b[A-Za-z]+\b"
        science_terms = re.findall(science_term_pattern, clue)

        # Pattern for space-related terms (e.g., rockets, gravity, satellites)
        space_term_pattern = r"\b[A-Za-z]+\b"
        space_terms = re.findall(space_term_pattern, clue)

        # Pattern for celestial bodies or phenomena (e.g., stars, cosmos)
        celestial_pattern = r"\b[A-Za-z]+\b"
        celestial_terms = re.findall(celestial_pattern, clue)

        if chemicals or science_terms or space_terms or celestial_terms:
            patterns.append(
                {
                    "clue": clue,
                    "chemicals": chemicals,
                    "science_terms": science_terms,
                    "space_terms": space_terms,
                    "celestial_terms": celestial_terms,
                }
            )

    return patterns


# Detect patterns in 'sci.med' and 'sci.space' clues
scimed_patterns = detect_patterns(scimed_clues_list)
scispace_patterns = detect_patterns(scispace_clues_list)

print("Detected patterns in 'sci.med' clues:")
for pattern in scimed_patterns:
    print(f"Clue: {pattern['clue']}")
    if pattern["chemicals"]:
        print(f"Chemicals: {pattern['chemicals']}")
    if pattern["science_terms"]:
        print(f"Science Terms: {pattern['science_terms']}")
    print("----------------")

print("Detected patterns in 'sci.space' clues:")
for pattern in scispace_patterns:
    print(f"Clue: {pattern['clue']}")
    if pattern["space_terms"]:
        print(f"Space Terms: {pattern['space_terms']}")
    if pattern["celestial_terms"]:
        print(f"Celestial Terms: {pattern['celestial_terms']}")
    print("----------------")


# 
# # Technical Report: Text Treasure Hunt - The Vectorization Adventure
# 
# 1. Introduction
# 
# In this report, we embark on the journey of "The Vectorization Adventure," where we delve into the world of text processing and vectorization. Our mission is to decode clues, uncover hidden connections, and collaborate with other teams to reach the ultimate treasure. The adventure involves several stages, each requiring different techniques and approaches to progress towards the final goal.
# 
# 2. Setting Up
# 
# We start by setting up our environment, ensuring we have all the necessary libraries installed and importing them into our Python environment. The libraries include NLTK, Pandas, Scikit-learn, Gensim, and Spacy. Additionally, we provide an optional section for advanced exploration with Transformers, which requires installation and importation of the Transformers library.
# 
# 3.  Quest Begins - The Initial Clue
# 
# The first step is to decipher the initial clue provided by our instructor. We carefully analyze the clue to identify words or themes that stand out, which will guide us towards selecting the relevant topic category within the Newsgroup 20 dataset.
# 
# 4. Keyword Quest
# 
# With our initial clue in hand, we proceed to extract keywords using TF-IDF (Term Frequency-Inverse Document Frequency). We define a function extract_keywords that applies TF-IDF to the selected texts, extracting relevant keywords that may hint at the next topic or text to explore. The extracted keywords help illuminate our path forward.
# 
# 5. Semantic Safari
# 
# Next, we embark on a semantic analysis journey using word embeddings like Word2Vec or GloVe. We calculate similarities between our extracted keywords and texts in other categories, seeking unexpected connections that may lead us closer to the treasure. 
# Advanced exploration with Transformers is also provided for a deeper semantic understanding.
# 
# 6. Pattern Pursuit
# 
# In this stage, we search for unusual patterns within the texts using regular expressions. We provide examples of regular expressions to find potential codes or emails within the text data. By examining letter sequences, numbers, or other patterns, we may uncover hidden clues crucial for progressing in the adventure.
# 
# 7. Collaboration and Convergence
# 
# Teamwork is essential for success in this adventure. We discuss effective communication strategies for sharing findings and combining insights with other teams. Collaboration is key to solving the ultimate puzzle and locating the treasure by converging all the gathered clues.
# 
# 8. Reflection and Report
# 
# Finally, we reflect on our journey, documenting the methods, techniques, and insights gained at each stage. We provide detailed explanations of the code snippets used and their significance in progressing through the adventure. We discuss the most helpful text processing techniques, the empowering nature of vectorization, surprising discoveries made, and potential real-world applications of our skills.
# 
# In conclusion, "The Vectorization Adventure" is a challenging yet rewarding journey through the realm of text processing and vectorization. By carefully analyzing clues, extracting keywords, exploring semantic meanings, and uncovering patterns, we inch closer to the ultimate treasure. Through collaboration and reflection, we not only solve the adventure but also gain valuable insights and skills applicable in various real-world scenarios.
# 
# 
# 
# ### Reflection
# 
# Which text processing techniques were most helpful and why?
# - TF-IDF: Quickly identified the most relevant keywords, helping us zero in on the core themes.
# - Word Embeddings (Word2Vec): Showed us how words relate to each other, revealing hidden connections and synonyms.
# 
# How did vectorization empower you to find hidden connections?
# - Turning words into numbers allowed us to use mathematical tools to spot patterns and similarities that would be hard to see with just our eyes.
# 
# What was the most surprising part of this adventure?
# - I was amazed at how regular expressions uncovered a hidden code-like pattern – it felt like cracking a secret message!
# 
# How could you use these skills for other problems in the real world?
# - Sentiment Analysis: Understanding the positive or negative "tone" of customer reviews or social media posts.
# - Topic Modeling: Discovering the underlying themes in large collections of news articles or other texts.
# - Better Search Engines: Creating search tools that find results based on the meaning of words, not just exact matches.
# 
# Thank you.

# ### End 
