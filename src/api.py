from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import coo_matrix,csr_matrix


import pandas as pd
import uvicorn
import joblib
import os


MODEL_PATH = 'models/als_tuned_model.pkl'
MAPPING_DIR = 'mapping'
INTERIM_PATH = 'data/interim'

class request_format(BaseModel) : 
    userid:int
    item_to_recommend: int = 10 
    


app = FastAPI()

@app.get("/")
def home():
    return "Recommender System in API."

def load_utility_matrix(user_mapping,artist_mapping) : 
    user_artist_path = os.path.join(INTERIM_PATH,"interim_user_artists.csv")
    user_artist_df = pd.read_csv(user_artist_path)
    user_artist_df.userID = user_artist_df.userID.map(user_mapping)
    user_artist_df.artistID = user_artist_df.artistID.map(artist_mapping)
        
    row = user_artist_df.userID.values
    col = user_artist_df.artistID.values
    data = user_artist_df.weight.values


    implicit_utility = coo_matrix((data,(row,col)))
    implicit_utility = implicit_utility.tocsr()
    return implicit_utility

def load_artist_data() : 
    artist_path = os.path.join(INTERIM_PATH,"interim_artists.csv")
    artist_data = pd.read_csv(artist_path,usecols=['id','name'])
    return artist_data

@app.post("/recommend/")
def recommend_artist_to_user(data: request_format):    
    #check user if exists 
    #load reference first, userid to ordered id 
    user_id_to_ordered_id = joblib.load(os.path.join(MAPPING_DIR,"user_id_to_ordered_id.pkl"))
    ordered_id_to_user_id = joblib.load(os.path.join(MAPPING_DIR,"ordered_id_to_user_id.pkl"))
    artist_id_to_ordered_id = joblib.load(os.path.join(MAPPING_DIR,"artist_id_to_ordered_id.pkl"))
    ordered_id_to_artist_id = joblib.load(os.path.join(MAPPING_DIR,"ordered_id_to_artist_id.pkl"))
  
    #check in keys 
    if data.userid not in  list(user_id_to_ordered_id.keys()) : 
        raise ValueError(f'User ID : {data.userid} Not Found.')
    
    user_ordered_id = user_id_to_ordered_id[data.userid]
    
    #load model 
    als_model = joblib.load(MODEL_PATH)
    
    #load utility_matrix
    utility_matrix = load_utility_matrix(user_mapping=user_id_to_ordered_id,
                                         artist_mapping=artist_id_to_ordered_id)

    #recommend user 
    recommendations = als_model.recommend(userid=user_ordered_id,
                        user_items=utility_matrix[user_ordered_id],
                        N=data.item_to_recommend)
    
    artists_ordered_ids = recommendations[0]
    #convert itemids --> to unordered
    artists_id = [ordered_id_to_artist_id[id] for id in artists_ordered_ids ]
    
    #retrieve artist name 
    artist_data = load_artist_data()
    
    artist_name = artist_data.loc[artist_data['id'].isin(artists_id),'name'].tolist()
    
    return {"recommended_artist" : artist_name, }
    
if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)