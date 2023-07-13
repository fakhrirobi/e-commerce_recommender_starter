from fastapi import FastAPI
from pydantic import BaseModel
from scipy.sparse import coo_matrix,csr_matrix


import pandas as pd
import uvicorn
import joblib
import os

from utils import load_config



MODEL_PATH = 'models/als_tuned_model.pkl'
MAPPING_DIR = 'mapping'
INTERIM_PATH = 'data/interim'

class request_format(BaseModel) : 
    reviewerid:int
    item_to_recommend: int = 10 
    


app = FastAPI()

@app.get("/")
def home():
    return "Recommender System in API."




def load_utility_data(path) : 
    data = pd.read_csv(path)
    return data 

def load_mapper(config) :
    reviewer_id_to_ordered_id = joblib.load(config['reviewer_id_to_ordered_id_path'])
    ordered_id_to_reviewer_id = joblib.load(config['ordered_id_to_reviewer_id_path'])

    item_id_to_ordered_id = joblib.load(config['item_id_to_ordered_id_path'])
    ordered_id_to_item_id = joblib.load(config['ordered_id_to_item_id_path'])
    
    return reviewer_id_to_ordered_id,ordered_id_to_reviewer_id,item_id_to_ordered_id,ordered_id_to_item_id
    
def find_unrated_items(userid,utility_data) : 

    
    sliced_utility = utility_data.loc[utility_data['reviewerID']==userid,:]
    #extract consumed item 

    total_item = set(utility_data.itemID.unique())
    user_consumed_item = set(sliced_utility.itemID.unique())
    
    unconsumed_item = list(total_item.difference(user_consumed_item))
    return unconsumed_item




def predict_on_reviewer_id(userid,item_list,model,n_items=5) : 
    prediction_df = pd.DataFrame()
    
    predictions = []
    for item in item_list : 
        prediction = model.predict(uid=userid,iid=item)
        predictions.append(prediction[3])
    prediction_df['itemID']= item_list    
    prediction_df['predicted_ratings'] = predictions
    # prediction_df['reviewer_id'] = userid
    
    prediction_df = prediction_df.sort_values('predicted_ratings',ascending=False)
    prediction_df = prediction_df.head(n_items)
    return prediction_df['itemID'].values.tolist()
    
    
@app.post("/recommend/")
def recommend_artist_to_user(data: request_format):    
    
    #load config data 
    config= load_config()
    #load user reference 
    
    (reviewer_id_to_ordered_id,
    ordered_id_to_reviewer_id,item_id_to_ordered_id,ordered_id_to_item_id) = load_mapper(config=config)

  
    #check in keys 
    # if data.reviewerid not in  list(reviewer_id_to_ordered_id.keys()) : 
    #     raise ValueError(f'User ID : {data.reviewerid} Not Found.')
    
    # user_ordered_id = reviewer_id_to_ordered_id[data.reviewerid]
    
    trained_model = joblib.load(config['best_model_path'])
    
    utility_data = load_utility_data(path=config['utility_data_path'])
    
    #unique id 
    
    unconsumed_item = find_unrated_items(userid=data.reviewerid,utility_data=utility_data)
    
    predicted_items = predict_on_reviewer_id(userid=data.reviewerid,
                                           item_list=unconsumed_item,
                                           model=trained_model,n_items=data.item_to_recommend)

    prediction = {'Recommendation for userid : ' :predicted_items }
    return prediction
    
    
    
if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)