from anaSALIMI import *

commands = {}

def command(func):
    commands[func.__name__] = func
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


song1 = get_song_by_title("Hey Soul Sister")

print(song1)

@command
def similar_songs(*target_song_name: str) -> list:
    
    target_song = get_song_by_title(" ".join(target_song_name))
    
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
     
    print(similar_songs_list)



while True:
    inp = input("$ ")
    inp = inp.split(" ")
    if len(inp) == 1:
        commands[inp[0]]()
    else:
        commands[inp[0]](*inp[1:])