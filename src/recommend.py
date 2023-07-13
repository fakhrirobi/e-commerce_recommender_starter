import joblib 
import os
import pandas as pd
import argparse
from utils import load_config


def initiliaze_argparse() : 
    parser = argparse.ArgumentParser(description='Process some integers.')
    
    parser.add_argument('--userid')
    
    return parser



    

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
    return prediction_df



def load_utility_data(path) : 
    data = pd.read_csv(path)
    return data 

if __name__ == "__main__" : 
    parser = initiliaze_argparse()
    args = parser.parse_args()
    userid = int(args.userid)
    
    CONFIG_DATA = load_config()
    
    #load model 
    model_path = os.path.join(CONFIG_DATA['best_model_path'],'best_model_svd.pkl')
    
    trained_model = joblib.load(model_path)
    

    
    #cara prediksi untuk semua item adalah dengan looping semua iid 
    utility_data = load_utility_data(path=CONFIG_DATA['utility_data_path'])
    
    #unique id 
    
    unconsumed_item = find_unrated_items(userid=userid,utility_data=utility_data)
    prediction_df = predict_on_reviewer_id(userid=userid,
                                           item_list=unconsumed_item,
                                           model=trained_model,n_items=10)
    print('Recommendation for userid : ',userid)
    print(prediction_df)
    