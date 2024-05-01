#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Text Sentiment Analysis
import pandas as pd
import numpy as np
import re
import nltk
nltk.download('punkt')
from nrclex import NRCLex
import string
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sentence_transformers import SentenceTransformer

from torch.nn.functional import cosine_similarity as torch_cosine_similarity


import requests
import os

# book_info = pd.read_csv('app/Book_description.csv')
# music_info = pd.read_csv('app/songs-dataset.csv')


def get_summary1(book_name, author_name):
    api_key = os.getenv('GOOGLE_BOOKS_API_KEY')
    url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{book_name}+inauthor:{author_name}&key={api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if 'items' in data and len(data['items']) > 0:
            item = data['items'][0]
            summ1 = item['volumeInfo'].get('description', 'Summary not available.')
            return summ1
    return ''

def get_summary2(book_name, author_name):
    book_info = pd.read_csv('app/book_summary.csv')

    filtered_row = book_info[(book_info['title'] == book_name) & (book_info['author'] == author_name)]
    if len(filtered_row)>0:
        return filtered_row['summary'].iloc[0]
    
    return ''

def get_summary3(book_name, author_name):
    book_info = pd.read_csv('app/Book_description.csv')
    book_name = book_name.lower()
    author_name = author_name.lower()

    #filtered_row = book_info[(book_info['title'].str.lower() == book_name) & (book_info['author'].str.lower().str.contains(author_name))]
    filtered_row = book_info[(book_info['title'].str.lower() == book_name) & (book_info['author'].str.lower().str.contains(author_name, regex=False))]

    if len(filtered_row)>0:
        return filtered_row['description'].iloc[0]
    
    return ''

def get_book_summaries(book_name, author_name):

    summ1 = preprocess_text(get_summary1(book_name, author_name))
    summ2 = preprocess_text(get_summary2(book_name, author_name))
    summ3 = preprocess_text(get_summary3(book_name, author_name))
    #print('book ', book_name)
    
    return summ1+summ2

def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub('\n+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text
    return ''

def find_emotion_scores(description):
    emotion = NRCLex(description)
    raw_score = emotion.raw_emotion_scores
    mean_score=0
    if len(raw_score) > 1:
        mean_score = sum(raw_score.values()) / len(raw_score)
        std_dev = (sum((x - mean_score) ** 2 for x in raw_score.values()) / len(raw_score)) ** 0.5
        
    else:
        std_dev = 1  # Set std_dev to a non-zero value to avoid division by zero

    if std_dev == 0:
        normalized_score = {emotion: 0 for emotion, score in raw_score.items()}
    else:
        normalized_score = {emotion: (score - mean_score) / std_dev for emotion, score in raw_score.items()}

    if normalized_score:  # Check if the dictionary is not empty
        max_score = max(normalized_score.values())
        min_score = min(normalized_score.values())
    else:
        max_score, min_score = 0, 0

    if max_score != min_score:
        normalized_score = {emotion: (score - min_score) / (max_score - min_score) for emotion, score in normalized_score.items()}
    else:
        normalized_score = {emotion: 0 for emotion, score in raw_score.items()}

    formatted_score = {emotion: format(score, '.4f') for emotion, score in normalized_score.items()}
    return formatted_score

def calculate_normalized_lyrics():
    results_lyrics = []
    results_lyrics_score = []
    music_info = pd.read_csv('app/songs-dataset.csv')

    music_lyrics = music_info['lyrics'].astype(str).apply(lambda x: x.lower())
    music_lyrics = music_lyrics.str.replace('\n',' ')

    for lyrics in music_lyrics:
        emotion_scores = find_emotion_scores(lyrics)
        results_lyrics_score.append(emotion_scores) 

    #lyrics to dataframe
    title = pd.DataFrame(music_info['track_name'])
    artist = pd.DataFrame(music_info['track_artist'])
    lyrics = pd.DataFrame(music_info['lyrics'])
    #id_df = pd.DataFrame(results_lyrics)
    score_df = pd.DataFrame(results_lyrics_score)
    merged_df = pd.concat([title, artist, lyrics, score_df], axis=1)   

    results_lyrics_df = pd.DataFrame(merged_df)
    results_lyrics_df.to_csv('NormalizationLyricsScore.csv', index=False) 
    return results_lyrics_df          



# def calculate_normalized_book_description():
#     results_descriptions = []
#     results_descriptions_score = []
#     book_info = pd.read_csv('app/Book_description.csv')
#     for _, row in book_info.iterrows():
#         title = row['title']
#         author = row['author']
#         book_description = row['description']
#         #book_description += get_book_summaries(title, author)


#         emotion_scores = find_emotion_scores(book_description)

#         results_descriptions.append(title)
#         results_descriptions_score.append(emotion_scores)

#     # book descriptions to dataframe
#     genres = pd.DataFrame(book_info["genres"])
#     id_df2 = pd.DataFrame(results_descriptions)
#     score_df2 = pd.DataFrame(results_descriptions_score)
#     merged_df2 = pd.concat([id_df2,genres, score_df2], axis=1)   

#     # book descriptions to csv
#     results_description_df = pd.DataFrame(merged_df2)
#     results_description_df.rename(columns={results_description_df.columns[0]: 'book'}, inplace=True)
#     results_description_df.to_csv('NormalizationDescriptionScore.csv', index=False)
    
#     return results_description_df 


# results_lyrics_df = calculate_normalized_lyrics()
# results_description_df = calculate_normalized_book_description()
# results_lyrics_df = results_lyrics_df.drop(['negative', 'positive'], axis=1)

# def fetch_lyrics(book_name, author_name, results_description_df):
    
#     try:

#         book_name_lower = book_name.lower()
#         book_row = results_description_df[results_description_df['book'].str.lower().str.contains(book_name_lower)].head(1)
        
#         if book_row.empty:
#             #add the book details in Book_description dataset
#             summary = get_book_summaries(book_name, author_name)
#             if summary == '':
#                 return None
#             book_info = pd.read_csv('app/Book_description.csv')
            
#             book_info.loc[len(book_info)] = [book_name, author_name, summary, '']
#             book_info.to_csv('app/Book_description.csv', index=False)
#             print('saved book info')
#             #find emotion scores
#             emotion_scores = find_emotion_scores(summary)

#             #add this book in normalized book description dataset
#             normalized_book = pd.read_csv('NormalizationDescriptionScore.csv')
#             normalized_book.loc[len(normalized_book)] = [book_name, '', emotion_scores]
#             normalized_book.to_csv('NormalizationDescriptionScore.csv', index=False)

#             #get row to process 
#             book_row = normalized_book[(normalized_book['book'].str.lower() == book_name_lower)]
#             #book_row = book_row.drop(['negative', 'positive'], axis=1)

#         return book_row
#     except Exception as e:
        # print(e)
        # return None

def calculate_cosine_similarity(book_vector, song_vectors):
    book_vector = np.nan_to_num(book_vector, nan=0)
    song_vectors = np.nan_to_num(song_vectors, nan=0)
    similarities = sklearn_cosine_similarity(book_vector.reshape(1, -1), song_vectors)
    return similarities.flatten()

def fetch_lyrics_and_calculate_similarity(book_name, author_name, n, results_lyrics_df):
    
    try:
        
        # book_row = fetch_lyrics(book_name, author_name, results_description_df)
        # if book_row is None or book_row.empty:
        #     return results_lyrics_df 
        # book_summary = get_book_summaries(book_name, author_name)
        # book_emotion_scores = find_emotion_scores(book_summary)
        
        # emotions = ['anticipation', 'disgust', 'joy', 'sadness', 'surprise', 'trust', 'anger', 'fear']

        # # Ensure the emotions columns match between book_row and lyrics_with_max_emotion
        # common_emotions = list(set(emotions) & set(book_emotion_scores.keys()))
        # # book_emotions = book_emotion_scores[common_emotions].fillna(0).values.flatten().astype(float)
        # book_emotions = np.array([book_emotion_scores[emotion] for emotion in common_emotions], dtype=float)
        # song_emotions = results_lyrics_df[common_emotions].fillna(0).values.astype(float)

        book_summary = get_book_summaries(book_name, author_name)
        book_emotion_scores = find_emotion_scores(book_summary)

        emotions = ['anticipation', 'disgust', 'joy', 'sadness', 'surprise', 'trust', 'anger', 'fear']

        # Ensure the emotions columns match between book_row and lyrics_with_max_emotion
        common_emotions = list(set(emotions) & set(book_emotion_scores.keys()))

        # Extract emotion scores for common emotions and convert them into a NumPy array
        book_emotions = np.array([book_emotion_scores[emotion] for emotion in common_emotions], dtype=float)
        song_emotions = results_lyrics_df[common_emotions].fillna(0).values.astype(float)

 
        similarities = calculate_cosine_similarity(book_emotions, song_emotions)

        results_lyrics_df['cos_similarity'] = similarities

        most_similar_songs = results_lyrics_df.sort_values(by='cos_similarity', ascending=False).head(n)
        print('here')
        return most_similar_songs

    except Exception as e:
        print(e)
        return results_lyrics_df

def load_transformer_model():
    model_name = 'paraphrase-MiniLM-L6-v2'
    model = SentenceTransformer(model_name)
    return model

def rank_songs(book_name, author_name, n, results_lyrics_df):

    df = fetch_lyrics_and_calculate_similarity(book_name, author_name, 50, results_lyrics_df)

    lyrics = df['lyrics'].tolist()

    model = load_transformer_model()

    lyric_embeddings = model.encode(lyrics, convert_to_tensor=True)

    all_summaries = get_book_summaries(book_name, author_name)

    query = preprocess_text(book_name + all_summaries)

    query_embedding = model.encode(query, convert_to_tensor=True)

    similarities = torch_cosine_similarity(query_embedding, lyric_embeddings)

    sorted_indices = similarities.argsort(descending=True)

    top_n_indices = sorted_indices[:n].detach().cpu().numpy() # Convert PyTorch tensor to numpy array

    tops_songs_df = df.iloc[top_n_indices]
    return tops_songs_df    
