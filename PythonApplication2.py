import os
import sys
import json
import spotipy
import webbrowser
import spotipy.util as util
from json.decoder import JSONDecodeError
import itunespy
import MySQLdb
import csv
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

song_count=1
UserID: "kzip1mgyp15anlwvfdrpf466u?si=rniExKUQSBi1P6aKcOF4FA"
client_id='c11c83e201bc4c23ae8e09c86d98129f'
secret='3ce325660f554a8e9de0a61212a74d68'
username='kzip1mgyp15anlwvfdrpf466u?si=rniExKUQSBi1P6aKcOF4FA'
redirect='http://google.com/'
scope=None
try:
    token = util.prompt_for_user_token(username, scope, client_id, secret, redirect)
except:
    os.remove(f".cache-{username}")
    token=util.prompt_for_user_token(username)
spotifyObject=spotipy.Spotify(auth=token)
user=spotifyObject.current_user()

def train_model():
    song=pd.read_csv('songs2.csv',delimiter=",",encoding='latin')
    
    loudness = song[['loudness']].values
    min_max_scaler = preprocessing.MinMaxScaler()
    loudness= min_max_scaler.fit_transform(loudness)
    loudness=loudness.flatten()
   
    loudness=pd.Series(loudness)
    song['loudness']=loudness
    
    features=['acousticness','danceability','liveness','loudness','speechiness']#,'valence','energy']
    #features=['valence','energy']
    X=song.loc[:,features]
    #To find elbow for k-means, Chose k=4... cuz why not
    '''
    sum=[]
    K=range(1,15)
    for k in K:
        km=KMeans(n_clusters=k)
        km=km.fit(X)
        sum.append(km.inertia_)
    plt.figure(figsize=(16,8))
    plt.plot(K,sum,'bx-')
    plt.show()'''

    km=KMeans(n_clusters=4)
    km.fit(X)
    labels=km.labels_
    centroids=km.cluster_centers_

    #created a file named model_pickle and stored the trained model in it, which can be used to make future predictions
    with open('model_pickle','wb') as file:
        pickle.dump(km,file)
    with open('scale_pickle','wb') as file:
        pickle.dump(min_max_scaler,file)
    #created a csv file to store the data and the cluster labels
    song["Cluster"]=labels
    #print(song)
    song.to_csv("song3.csv",index=False)
    song.to_csv("song4.csv",index=False,columns=["Sno","Song","Artist","ArtistID","Cluster"])

def f(artist,track):
    track_id=spotifyObject.search(q='artist:'+artist+' track:'+track,type='track')
    if(len(track_id["tracks"]["items"])!=0):
        #print(json.dumps(track_id,sort_keys=True,indent=4))
        track_id=track_id["tracks"]["items"][0]["id"]
        #print(track_id)
        #print("Audio features")
        res=spotifyObject.audio_features(track_id)
        #print(json.dumps(res,sort_keys=True,indent=4))
        features={"acousticness":res[0]["acousticness"],"danceability":res[0]["danceability"],"liveness":res[0]["liveness"],"loudness":res[0]["loudness"],"speechiness":res[0]["speechiness"],"valence":res[0]["valence"],"energy":res[0]["energy"]}
        return features
        #res=spotifyObject.audio_analysis(track_id)
    else:
        return 'Null'

def Get_albums(artist):
    #warnings.filterwarnings("ignore")
    #albums=PyLyrics.getAlbums(singer=artist)
    #return albums
    try:
        a=itunespy.search_artist(artist)
        albums=a[0].get_albums()
        Albums=[]
        c=0
        for i in albums:
            Albums.append(i.collection_name)
            if c>=2:                                #For now, the max albums for an artist is 3
                break
            c+=1
        return Albums
    except:
        return('Null')

def Get_songs(Album):
    try:
        album = itunespy.search_album(Album)  # Returns a list
        tracks = album[0].get_tracks()  # Get tracks from the first result
        t=[]
        c=0
        for track in tracks:
            t.append(track.track_name)
            if c>=2:                            #For now, the max songs from each album is 5
                break
            c+=1
        return t
    except:
        return 'Error'

def GetArtistID(a):
    s="select id from artist where name = '"+a+"';"
    mydb=MySQLdb.connect(host="localhost",user="root",passwd="xj0461",database="project2")
    mycursor=mydb.cursor()
    mycursor.execute(s)
    myresult = mycursor.fetchall()
    for i in myresult:
        for j in i:
            return j

def write_song_to_csv(artist,artistID):               #write_song writes song features into csv file songs2.csv
    global song_count
    album=Get_albums(artist)
    if(album!="Null"):
        songs=[]
        for i in album:
            songs=[]
            res = Get_songs(i)
            if(res!='Error'):
                songs.extend(res)
                for j in songs:
                    x=f(artist,j)
                    if(x!="Null"):
                        with open('songs2.csv','a',newline='') as file:
                            writer=csv.writer(file)
                            writer.writerow([song_count,j,artist,artistID,x["acousticness"],x["danceability"],x["liveness"],x["loudness"],x["speechiness"],x["valence"],x["energy"]])
                            song_count+=1

def write_song():               #write_song authorizes spotify api creates file song.csv, reads artists from DB and call write_song_to_csv
    mydb=MySQLdb.connect(host="localhost",user="root",passwd="xj0461",database="project2")
    mycursor=mydb.cursor()
    statement="select name,id from artist"
    mycursor.execute(statement)
    myresult = mycursor.fetchall()
    #with open('songs2.csv','a',newline='') as file:
    #    writer=csv.writer(file)
    #    writer.writerow(["Sno","Song","Artist","ArtistID","acousticness","danceability","liveness","loudness","speechiness","valence","energy"])
    for i in myresult:
        write_song_to_csv(i[0],i[1])
        print(song_count)

def pred(features):
    fu=[features["acousticness"],features["danceability"],features["liveness"],features["loudness"],features["speechiness"]]
    with open('scale_pickle','rb') as file:
        min_max_scaler=pickle.load(file)
        fu[3]=min_max_scaler.transform(np.array([[fu[3]]]))
        fu[3]=fu[3][0,0]
        fu=np.array([fu])
        #print(fu)
    with open('model_pickle','rb') as file:
        model=pickle.load(file)
        c=(model.predict(fu))
        return(c[0])


def add_song(song,artist):
    data=pd.read_csv('song3.csv',usecols=[4,5,6,7,8,9,10])
    song_count=data.shape[0]+1
    x=f(artist,song)
    ArtID=GetArtistID(artist)
    if(x!='Null'):
        with open('song4.csv','a',newline='') as file:
            writer=csv.writer(file)
            fu=pred(x)
            writer.writerow([song_count,song,artist,ArtID,fu])
        song_count+=1
    else:
        print('Song or Artist could not be found')
#write_song()
#train_model()

def recommend(l):
    mydb=MySQLdb.connect(host="localhost",user="root",passwd="xj0461",database="project2")
    mycursor=mydb.cursor()
    for i in range(len(l)):
        statement="select s.name,a.name, s.cluster from songs s, artist a where s.ArtistID=a.ID and s.cluster= "+str(i)+" order by rand() limit "+str(l[i])+";"
        mycursor.execute(statement)
        myresult = mycursor.fetchall()
        for j in myresult:
            print(j)

'''a=input("Artist: ")
s=input("Song: ")
print(pred(f(a,s)))'''
#add_song('Before You Go','Lewis Capaldi')