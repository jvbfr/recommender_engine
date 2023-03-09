import numpy as np
import pandas as pd
from scipy.special import expit
from dataclasses import dataclass
from lightfm import LightFM
import config


@dataclass
class Recommendation:

    def train_recsys(self, data_recsys, user_features=None, item_features=None, epochs=30, num_threads=8, verbose=False):
        """
        It trains the recommendation algoritm
        
        Parameters
        -------------
        data_recsys: np.float32 coo_matrix of shape [n_users, n_items]
            The matrix containing user-item interactions.      
            
        user_features: np.float32 csr_matrix 
            A CSR matrix from Scipy with shape [n_users, n_user_features] (optional).
            
        item_features: np.float32 csr_matrix of shape [n_items, n_item_features] (optional) 
            A CSR matrix from Scipy with shape [n_users, n_user_features] (optional).
            
        epochs: int
            Number of epochs to run (optional).
        
        num_threads: int
            Number of parallel computation threads to use. Should not be higher than the number of physical cores (optional).
        
        verbose: bool
            Whether to print progress messages. If tqdm is installed, a progress bar will be displayed instead.
        Returns
        -------------
        model_recsys: lightfm.lightfm.LightFM
            The trained model.
        """
        model_recsys = LightFM(loss='warp', k=1)
        model_recsys.fit(
            data_recsys.get('interactions'), 
            user_features=user_features,
            item_features=item_features,
            epochs=epochs, 
            num_threads=num_threads,
            verbose=verbose
        )
        return model_recsys
    
    def _is_coldstart(self, user_id, user_reference_dict):
        """
        Check if the user is in cold-start state.
        Parameters
        ----------
        user_id : int or str
            The user identifier.
        user_reference_dict : dict
            A dictionary containing references to the users.
        Returns
        -------
        bool
            True if the user is in cold-start state, False otherwise.
        """
        return user_reference_dict.get(user_id) is None 

    def get_recommendation(self, user_index, model, list_item_lightfm_index):
        prediction = model.predict(
            user_ids = user_index, 
            item_ids = list_item_lightfm_index, 
            num_threads=config.NUM_THREADS
        ) 
        return prediction
        
    def create_data_recommendation(self, dask_recsys_prediction, dict_recsys_information):
        all_scores = dask_recsys_prediction.compute()
        score = all_scores.apply(np.max)
        item_id_recommended = all_scores.apply(lambda x: dict_recsys_information.get('inverse_item').get(np.argmax(x)))

        data_recommendation = pd.DataFrame(
            {
                'user_id': pd.Series(dict_recsys_information.get('inverse_user')),
                'item_id': item_id_recommended,
                'score': score
            }
        )
        return data_recommendation