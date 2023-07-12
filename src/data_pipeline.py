
import pandas as pd #data processing
import numpy as np 
import joblib #
import copy 


from utils import load_config
from surprise import Dataset, Reader 






def load_data(path) : 
    #read data
    data  = pd.read_csv(path)
    
    #print datashape
    print('Data shape : ',data.shape)
    
    
    #print datatypes 
    print('Data Types : \n ',data.dtypes)
    #return data 
    return data 
    
    
    
















def remove_min_interaction(rating_data,number_of_interaction,threshold=3) : 
    rating_data = rating_data.copy()
    filter_minimum_interaction = number_of_interaction['avg_interaction']<threshold
    under_threshold_reviewerID = number_of_interaction.loc[filter_minimum_interaction,'reviewerID'].tolist()
    rating_adjusted = rating_data.loc[
        ~rating['reviewerID'].isin(under_threshold_reviewerID) ]

    return rating_adjusted
    







def map_to_ordered_id(rating,config) : 
    rating = rating.copy()
    #map user id --> ordered id 
    reviewer_id_to_ordered_id = {}
    ordered_id_to_reviewer_id = {}
    for idx,reviewer_id in enumerate(rating['reviewerID'].unique()) : 
        reviewer_id_to_ordered_id[reviewer_id] = idx+1
        ordered_id_to_reviewer_id[idx+1] = reviewer_id




    #map user id --> ordered id 
    item_id_to_ordered_id = {}
    ordered_id_to_item_id = {}
    for idx,item_id in enumerate(rating['itemID'].unique()) : 
        item_id_to_ordered_id[item_id] = idx+1
        ordered_id_to_item_id[idx+1] = item_id
    
    rating.reviewerID = rating.reviewerID.map(reviewer_id_to_ordered_id)
    rating.itemID = rating.itemID.map(item_id_to_ordered_id)



    joblib.dump(reviewer_id_to_ordered_id,config['reviewer_id_to_ordered_id_path'])
    joblib.dump(ordered_id_to_reviewer_id,config['ordered_id_to_reviewer_id_path'])

    joblib.dump(item_id_to_ordered_id,config['item_id_to_ordered_id_path'])
    joblib.dump(ordered_id_to_item_id,config['ordered_id_to_item_id_path'])

    return rating


def train_test_split(utility_data, test_size, random_state):
    """
    Train test split the data
    ref: https://surprise.readthedocs.io/en/stable/FAQ.html#split-data-for-unbiased-estimation-py

    Parameters
    ----------
    utility_data : Surprise utility data
        The sample of whole data set

    test_size : float, default=0.2
        The test size

    random_state : int, default=42
        For reproducibility

    Returns
    -------
    full_data : Surprise utility data
        The new utility data

    train_data : Surprise format
        The train data

    test_data : Surprise format
        The test data
    """
    # Deep copy the utility_data
    full_data = copy.deepcopy(utility_data)

    # Generate random seed 
    np.random.seed(random_state)

    # Shuffle the raw_ratings for reproducibility
    raw_ratings = full_data.raw_ratings
    np.random.shuffle(raw_ratings)

    # Define the threshold
    threshold = int((1-test_size) * len(raw_ratings))
    
    # Split the data
    train_raw_ratings = raw_ratings[:threshold]
    test_raw_ratings = raw_ratings[threshold:]

    # Get the data
    full_data.raw_ratings = train_raw_ratings
    train_data = full_data.build_full_trainset()
    test_data = full_data.construct_testset(test_raw_ratings)

    return full_data, train_data, test_data











if __name__ == "__main__" : 
    
    CONFIG_DATA = load_config()
    
    rating = load_data(path=CONFIG_DATA['dataset_path'])
    
    number_of_interaction = (rating.groupby('reviewerID',as_index=False)
                            .agg(avg_interaction=pd.NamedAgg('itemID','count'))
                            )
    
    rating_data_filtered = remove_min_interaction(
                                                rating_data=rating,
                                                number_of_interaction=number_of_interaction,
                                                threshold=3)
    
    
    
    utility_data = map_to_ordered_id(rating=rating_data_filtered,config=CONFIG_DATA)
    
    reader = Reader(rating_scale=(1,5))
    
    utility_matrix = Dataset.load_from_df(
                    df = utility_data[['reviewerID', 'itemID', 'rating']].copy(),
                    reader = reader
                )

    full_data, train_data, test_data = train_test_split(utility_matrix,
                                                    test_size = 0.2,
                                                    random_state = CONFIG_DATA['seed'])
    
    
    joblib.dump(full_data,CONFIG_DATA['full_utility_matrix_path'])
    joblib.dump(train_data,CONFIG_DATA['train_utility_matrix_path'])
    joblib.dump(test_data,CONFIG_DATA['test_utility_matrix_path'])
    
    print('Successfully Saved Training,Test,and Full data')