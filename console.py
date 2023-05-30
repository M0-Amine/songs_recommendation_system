from main import *

commands = {}
def command(func):
    commands[func.__name__] = func
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

user1 = User("Khaoulitaa")
user2 = User("M. Amine")

print(f"Hello {user1.name}\n")
print("type 'help' to see the list of commands")

@command
def help():
    """ Displays this message """
    for key in commands:
        if key != "help":
            print(key, ":",  commands[key].__doc__)
    print("\n")

@command
def display_all_songs() -> None:
    """ 
    - Displays all the songs in the dataset
    """
    display_playlist(songs_list, "All the songs:")

@command
def same_genre(*genre: str):
    """
    - Displays a playlist of songs belonging to the same genre.
    - Args: genre (str): The genre of songs to retrieve.
    """
    genre = " ".join(genre)
    display_playlist(same_genre_songs(genre), f"{genre} songs")
  
@command
def same_artist(*artist: str):
    """
    - Displays a playlist of songs by the same artist.
    - Args: artist (str): The artist of songs to retrieve.
    """
    artist = " ".join(artist)
    display_playlist(same_artist_songs(artist), f"{artist}'s songs")

@command
def recommend_similar_songs(*target_song_name: str):
    """
    - Displays a list of songs similar to the target song using Euclidean distance.
    - Args: target_song_name (str): The target song title to find similar songs for.
    """
    target_song_name = " ".join(target_song_name)
    target_song = get_song_by_title(target_song_name)
    display_playlist(similar_songs(target_song), f"Recommended for you based on '{target_song_name} by {target_song.artist}'")

@command 
def play(*song_name: str):
    """
    - Adds a song to your listening history 
    - Args: song_name (str): The title of the song you want to played.
    """
    song_name = " ".join(song_name)
    song_id = get_song_by_title(song_name).get_song_id()
    user1.listen_to(song_id)
    print(f"'{song_name} by {get_song_by_id(song_id).artist}' is now playing.")

@command
def like(*song_name: str):
    """
    - Adds song to your Liked Songs.
    - Args: song_name (str): The title of the song you want to add to your Liked Songs.
    """
    song_name = " ".join(song_name)
    song_id = get_song_by_title(song_name).get_song_id()
    user1.add_to_liked_songs(song_id)
    print(f"'{song_name} by {get_song_by_id(song_id).artist}' was added to your Liked Songs")

@command
def add_genre(*genre: str):
    """
    - Adds a new genre to the list of preferred genres.
    - Args: genre (str): The genre to be added to the preferred genres list.
    """
    genre = " ".join(genre)
    user1.add_genre(genre)
    print(f"{genre} was added to your Preferred genres.")
    
@command
def recommend(*playlist_name: str):
    """
    - Displays a Playlist of newly recommended songs based on liked songs.
    - Args (Playlist): the playlist you want a recommendation based on. 
    """
    playlist_name = " ".join(playlist_name)
    playlist = user1.get_playlist_by_name(playlist_name)
    recom_list = user1.create_playlist(f"Recommended for you based on {playlist.name}")
    pl = [get_song_by_id(song) for song in playlist]    
    for song in pl:
        similar_songs = song.get_similar_songs()
        s1 = similar_songs[0]
        s2 = similar_songs[1]
        if not s1 in pl:
            recom_list.append(s1)
        if not s2 in pl:
            recom_list.append(s2)   
            
    print(recom_list)


while True:
    inp = input("$ ")
    inp = inp.split(" ")
    
    try:
        if len(inp) == 1:
            commands[inp[0]]()
        else:
            commands[inp[0]](*inp[1:])
    except Exception as e:
        print(e)  