from surprise import AlgoBase, KNNBasic, SVD
from surprise.model_selection import cross_validate
from surprise import accuracy
from utils import load_config
import joblib 


#custom model 
class MeanPrediction(AlgoBase):
    '''Baseline prediction. Return global mean as prediction'''
    def __init__(self):
        AlgoBase.__init__(self)

    def fit(self, trainset):
        '''Fit the train data'''
        AlgoBase.fit(self, trainset)

    def estimate(self, u, i):
        '''Perform the estimation/prediction.'''
        est = self.trainset.global_mean
        return est
    
def load_model_candidate() : 
    #instanciate model 
    model_baseline = MeanPrediction()
    model_baseline


    # Create Neighbor-based model -- K-Nearest Neighbor
    model_knn = KNNBasic(random_state=42)
    model_knn




    # Create matrix factorization model -- SVD-like
    model_svd = SVD(n_factors=100, random_state=42)
    
    model_candidate = {'baseline':model_baseline,
                       'knn' : model_knn, 
                       'svd':model_svd}
    
    return model_candidate

def load_all_data(config) :
    full_data = joblib.load(config['full_utility_matrix_path'])
    train_data = joblib.load(config['train_utility_matrix_path'])
    test_data = joblib.load(config['test_utility_matrix_path'])
    
    return full_data,train_data,test_data



def train_model(config) :
    full_data,train_data,test_data = load_all_data(config=config)
    model_candidate = load_model_candidate()
    
    model_score = {}
    for model_name in model_candidate.keys() :
        model = model_candidate[model_name]
        
        cv_model  = cross_validate(algo = model,
                            data = full_data,
                            cv = 5,
                            measures = ['rmse'])
        mean_rmse = cv_model['test_rmse'].mean()
        model_score[model_name] = mean_rmse 
    
    #find best model 
    best_score = 0 
    best_model = ''
    for model,score in model_score.items() : 
        print(f'Model : {model}, CV RMSE Score : {score}')
        if score > best_score : 
            best_model=model 
        else : 
            continue
    print(f'Best Model : {best_model}')
    
    model_best = model_candidate[best_model]
    
    #fit on full training data 
    model_best.fit(train_data)
    
    test_pred = model_best.test(test_data)
    test_rmse = accuracy.rmse(test_pred)
    
    print(f'Best Model : {best_model}, Final Eval Score : {test_rmse}')
    
    #save model as pickle object 
    folder = config['best_model_path']
    filename = f'{folder}best_model_{best_model}.pkl'
    
    joblib.dump(model_best,filename=filename)
    print('Model Saved')
    
    
if __name__ == "__main__" : 
    config = load_config()
    train_model(config=config)