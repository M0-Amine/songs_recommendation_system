import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# IDs generator
def id_gen():
    id = 0
    while True:
        id += 1
        yield id
generator = id_gen()

# Sorting algorithm
def selection_sort(arr):
    n = len(arr)
    for i in range(n - 1):
        min_index = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_index]:
                min_index = j
        arr[i], arr[min_index] = arr[min_index], arr[i]
        
# Playlist Displayer
pd.set_option('display.max_colwidth', None)
def display_playlist(iterable):
    """df = pd.Series(["" for i in range(len(iterable))], index=iterable)
    print("Playlist 1:")
    display(df)"""
    print('Playlist 1:\n')
    for song in iterable:
        print(song)
 
# List to DynamicArray converter
def Dynamic_Array(lst: list):
    result = DynamicArray()
    for elem in lst:
        result.append(elem)
    return result
 
 
 
 ### Displaying data frame:
songs_dataset_json = '/home/amine/Desktop/LBDII/LBDProjects/songs_recommendation_system/main_folder/music_dataset.json'
df_song = pd.read_json(songs_dataset_json)
df_song_ = df_song.style.set_properties(**{'text-align': 'left'})


### Scaling:
df_song_scaled = df_song.copy()

# Identify the columns we wish to scale
columns_to_normalize = ["BeatsPerMinute", "Energy",	"Danceability",	"Loudness/dB",	"Liveness",	"Valence",	"Acousticness",	"Speechiness",	"Popularity"]

# Scaling the columns corresponding to the song features
scaler = MinMaxScaler()
df_song_scaled[columns_to_normalize] = scaler.fit_transform(df_song_scaled[columns_to_normalize])

# Output the scaled DataFrame
df_song_scaled_ = df_song_scaled.style.set_properties(**{'text-align': 'left'})
#df_song_scaled_



class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class DynamicArray:
    
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def is_empty(self) -> bool:
        return self.size == 0

    def __len__(self) -> int:
        return self.size

    def append(self, value) -> None:
        new_node = Node(value)

        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

        self.size += 1

    def set(self, index, value) -> None:
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")

        current = self.head
        for _ in range(index):
            current = current.next

        current.value = value

    def remove(self, index) -> None:
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")

        if index == 0:
            self.head = self.head.next
            if self.head is None:
                self.tail = None
        else:
            current = self.head
            for _ in range(index - 1):
                current = current.next

            current.next = current.next.next
            if current.next is None:
                self.tail = current
        self.size -= 1
        
    def __iter__(self) -> None:
        current = self.head
        while current:
            yield current.value
            current = current.next
            
    def __repr__(self) -> str:
        string = ""
        for elem in self:
            string += f"{elem}"
        
        return string
    
    def __getitem__(self, key):
        
        if isinstance(key, slice):
            result = DynamicArray()
            start=key.start
            stop=key.stop
            step=key.step
            
            if step is None:
                step=1  
                
            if start is None:
                start=0
                
            if stop is None:
                stop=len(self)  
            
            if start > stop:
                return result
            
            if start < 0 or stop < 0:
                raise IndexError ("Negative indexing is not supported")
        
            for ind in range(start,stop, step):
                result.append(self[ind])
                    
            return result
        
        else: 
            if key < 0 or key >= self.size:
                raise IndexError("Index out of range")

            current = self.head
            for _ in range(key):
                current = current.next

            return current.value

    def __list__(self) -> list:
        lst = []
        for song in self:
            lst.append(song)
        return lst
    
    def copy(self):
        return self[:]
    
    
    
class Vector_10_dim:
    def __init__(self) -> None:
        self.x1 = 0
        self.x2 = 0
        self.x3 = 0
        self.x4 = 0
        self.x5 = 0
        self.x6 = 0
        self.x7 = 0
        self.x8 = 0
        self.x9 = 0

    def set_vector_components(self, x1, x2, x3, x4, x5, x6, x7, x8, x9) -> None:
        self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7, self.x8, self.x9 = x1,x2,x3,x4,x5,x6,x7,x8,x9
        
        
        

class Song(Vector_10_dim):
    
    def __init__(self, title: str, artist: str, genre: str) -> None:
        super().__init__()
        self.id = next(generator)
        self.title = title
        self.artist = artist
        self.genre = genre
        
    def get_similar_songs(self) -> DynamicArray:
        """
    Returns a DynamicArray of similar songs based on the features specified previously.
    
    Returns:
        DynamicArray: A DynamicArray containing the similar songs.
    """
        return similar_songs(self)
    
    def get_same_genre_songs(self) -> DynamicArray:
        """
    Returns a DynamicArray of songs belonging to the same genre as the current song.
    
    Returns:
        DynamicArray: A DynamicArray containing the songs of the same genre.
    """
        ind = -1
        t = same_genre_songs(self.genre)
        for song in t:
            ind += 1
            if song == self:
                break
        
        t.remove(ind)
        return t
    
    def get_same_artist_songs(self) -> DynamicArray:
        """
    Returns a DynamicArray of songs by the same artist as the current song.
    
    Returns:
        DynamicArray: A DynamicArray containing the songs by the same artist.
    """
        ind = -1
        t = same_artist_songs(self.artist)
        for song in t:
            ind += 1
            if song == self:
                break
        
        t.remove(ind)
        return t
    
    def __repr__(self) -> str:
        return f"{self.title} - {self.artist}"
   
   
   
   
   # Extracting data & Creating Song instances
features = ["BeatsPerMinute", "Energy",	"Danceability",	"Loudness/dB",	"Liveness",	"Valence",	"Acousticness", "Speechiness",	"Popularity"]
songs_list = DynamicArray()
genres_list = []

for i in range(583):
    songs_list.append(Song(df_song['Title'][i], df_song['Artist'][i], df_song['Genre'][i]))
    genres_list.append(df_song['Genre'][i])

genres_list = list(set(genres_list))

for i, song in enumerate(songs_list):
    song.set_vector_components(*(df_song[feature][i] for feature in features))

# Functions for Song class instances
def get_song_by_id(song_id: int) -> Song:
    """
    Retrieves a song by its ID from the songs list.
    
    Args:
        song_id (int): The ID of the song to retrieve.
    
    Returns:
        Song: The song object corresponding to the given ID.
    """
    for song in songs_list:
        if song.id == song_id:
            return song
        
def get_song_by_title(title: str) -> Song:
    """
    Retrieves a song by its title from the songs list.
    
    Args:
        title (str): The title of the song to retrieve.
    
    Returns:
        Song: The song object corresponding to the given title.
    """
    for song in songs_list:
        if song.title == title:
            return song
        
def Euclidean_distance(song1: Song, song2: Song) -> float:
    """
    Computes the Euclidean distance between two songs.
    
    Args:
        song1 (Song): The first song object.
        song2 (Song): The second song object.
    
    Returns:
        float: The Euclidean distance between the two songs.
    """

    point1 = np.array((song1.x1, song1.x2, song1.x3, song1.x4, song1.x5, song1.x6, song1.x7, song1.x8, song1.x9))
    point2 = np.array((song2.x1, song2.x2, song2.x3, song2.x4, song2.x5, song2.x6, song2.x7, song2.x8, song2.x9))
    
    return np.linalg.norm(point1 - point2)

def similar_songs(target_song: Song) -> list:
    """
    Returns a list of songs similar to the target song based on Euclidean distance.
    
    Args:
        target_song (Song): The target song object to find similar songs for.
    
    Returns:
        list: A list of song objects similar to the target song.
    """
    distances_list = []
    similar_songs_list = []
    
    for other_song in songs_list:
        if not other_song is target_song:
            distance = Euclidean_distance(target_song, other_song)
            distances_list.append(distance)
        
    distances_ordred_list = sorted(enumerate(distances_list), key=lambda x: x[1])
    
    for index, distance in distances_ordred_list:
        similar_songs_list.append(songs_list[index])
        
    similar_songs_list = similar_songs_list[:6]
     
    return similar_songs_list

def same_genre_songs(genre: str) -> DynamicArray:
    """
    Returns a DynamicArray of songs belonging to the same genre.
    
    Args:
        genre (str): The genre of songs to retrieve.
    
    Returns:
        DynamicArray: A DynamicArray containing the songs of the specified genre.
    """
    same_genre = DynamicArray()
    for song in songs_list:
        if song.genre == genre:
            same_genre.append(song)
            
    return same_genre

def same_artist_songs(artist: str) -> DynamicArray:
    """
    Returns a DynamicArray of songs by the same artist.
    
    Args:
        artist (str): The artist of songs to retrieve.
    
    Returns:
        DynamicArray: A DynamicArray containing the songs by the specified artist.
    """
    same_artist = DynamicArray()
    for song in songs_list:
        if song.artist == artist:
            same_artist.append(song)
            
    return same_artist

title_list = [song.title for song in songs_list]
def delete_occurrences(lst: DynamicArray):
    """
    Removes duplicate occurrences of songs in the provided DynamicArray.
    
    Args:
        lst (DynamicArray): The DynamicArray containing songs.
    
    Returns:
        list: A list of songs with duplicate occurrences removed.
    """
    result = []
    length = len(lst)
    lst = list(lst)

    for i in range(length):
        if lst[i].title not in title_list[i+1:]:
            result.append(lst[i])

    return result



class Playlist(DynamicArray):
    
    def __init__(self, name="Playlist"):
        super().__init__()
        self.name = name
    
    def display(self):
        """
    Converts the DynamicArray elements into a string representation for display.
    
    Returns:
        str: A string representation of the DynamicArray elements joined by '|'.
    """
        string = ""
        list_s = []
        
        for elem in self:
            list_s.append(str(elem))
            
        string += " | ".join(list_s)
        
        return string
    
    def __repr__(self) -> str:
        return self.display()
    
    


class User:
    
    def __init__(self, name: str) -> None:
        self.name = name
        
        self.liked_songs = Playlist("liked_songs")
        self.lisetening_history = {}
        self.preferred_genres = Playlist("preferred genres")
        self.following = Playlist("following")
        
    def listen_to(self, song_id: int) -> None:
        """
    Adds a song to the listening history and increments the listening count of the song by one.
    
    Args:
        song_id (int): The ID of the song to be added to the listening history.
    
    Returns:
        None
        """
        if song_id in self.listening_history:
            self.listening_history[song_id][0] += 1
        else:
            self.listening_history[song_id] = {'count': 1, 'rate': get_song_by_id(song_id) in self.liked_songs}
        
    def add_to_liked_songs(self, song_id: int) -> None:
        """
        Adds a song to the liked_songs list.
        
        Args:
            song_id (int): The ID of the song to be added to the liked_songs list.
        
        Returns:
            None
        """
        if not song_id in self.liked_songs:
            self.liked_songs.append(song_id)
         
    def rate_song(self, song_id: int, rate: int) -> None:
        """
    Assigns a rating to a song in the listening history.
    
    Args:
        song_id (int): The ID of the song to be rated.
        rate (int): The rating value to assign to the song.
    
    Returns:
        None
    """
        if song_id in self.listening_history:
            self.listening_history[song_id]['rate'] = rate     
            
    def add_genre(self, genre: str) -> None:
        """
    Adds a new genre to the list of preferred genres.
    
    Args:
        genre (str): The genre to be added to the preferred genres list.
    
    Returns:
        None
    """
        if not genre in self.preferred_genres:
            self.preferred_genres.append(genre)
        
    def set_preferred_genres(self, genre: str) -> None:
        """
    Sets the preferred genres for the user.
    
    Args:
        genre (str): The genre to be set as a preferred genre.
    
    Returns:
        None
    """
        if not genre in genres_list:
            print(f"the genre {genre} is not found")
        else:
            if not genre in  self.preferred_genres:
                self.preferred_genres.append(genre)
           
    def liked_recommendation(self) -> DynamicArray:
        """
    Returns a DynamicArray of newly recommended songs based on liked songs.
    
    Returns:
        DynamicArray: A DynamicArray containing the newly recommended songs.
    """
        recom_list = DynamicArray()
        for song in self.liked_songs:
            similar_songs = song.get_similar_songs()
            s1 = similar_songs[0]
            s2 = similar_songs[1]
            if not s1 in self.liked_songs:
                recom_list.append(s1)
            if not s2 in self.liked_songs:
                recom_list.append(s2)   
                
        return recom_list
                    
    def history_recommendation(self) -> DynamicArray:
        """ returns a Dynamic Array of newly recommended songs based on listening_history """
        
        
        
        


