from flask import Flask, render_template, request, redirect, url_for
from RecommendSongs import rank_songs, calculate_normalized_lyrics 

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
from PIL import Image
import pandas as pd
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()
client_id = os.getenv('SPOTIFY_CLIENT_ID')
client_secret = os.getenv('SPOTIFY_CLIENT_SECRET')
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Note:  run if the csv files are not generated or Book_description and songs-dataset is changed
#results_lyrics_df = calculate_normalized_lyrics()
# results_description_df = calculate_normalized_book_description()
results_lyrics_df = pd.read_csv('NormalizationLyricsScore.csv')
# results_description_df = pd.read_csv('NormalizationDescriptionScore.csv')

results_lyrics_df = results_lyrics_df.drop(['negative', 'positive'], axis=1)

def get_album_cover(song_name, artist_name):
    # Search for the track
    results = sp.search(q=f'track:{song_name} artist:{artist_name}', limit=1)
    if results['tracks']['items']:
        track = results['tracks']['items'][0]
        # Get the album ID
        album_id = track['album']['id']
        # Get the album details
        album = sp.album(album_id)
        # Get the URL of the album cover
        image_url = album['images'][0]['url']
        return image_url  # Return the album cover URL
    else:
        return None  # Return None if track is not found
@app.route('/', methods=['GET', 'POST'])
def input():
    return render_template('input.html')
@app.route('/result', methods=['GET', 'POST'])
def result():
    book_name = request.args.get('book')
    author_name = request.args.get('author')

    print(f"Book Name: {book_name}, Author Name: {author_name}")
    #top_n_songs = fetch_lyrics_and_calculate_similarity(book_name, 50)
    songs = rank_songs(book_name, author_name, 20, results_lyrics_df)
    print(songs)
    '''
    if songs == 'Book not found in the dataset.':
        # Book not found in the dataset, render error template
        error_message = f"We don't have '{book_name}' in our dataset. Please try again with another book."
        return render_template('error.html', error_message=error_message)'''
    
    for index, song in songs.iterrows():
        track_name = song['track_name']
        track_artist = song['track_artist']
        
        # Get album cover URL for the song
        album_cover_url = get_album_cover(track_name, track_artist)
        
        # Update the 'album_cover_url' column in the DataFrame
        songs.at[index, 'album_cover_url'] = album_cover_url

    return render_template('result.html', songs=songs.to_dict('records'), book=book_name)
    
if __name__ == '__main__':
    app.run('0.0.0.0', port=8884, debug=True)