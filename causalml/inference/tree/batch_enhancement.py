import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from functools import reduce

class BatchEnhancementOnX:
    '''
    Used to get matched dataset
    '''
    def propensity_list_is_initialised(self):
        return self.ball_trees is not None

    def make_propensity_lists(self, train_ids, x, treatment_data, num_treatments):
        if x.shape[-1] > 200:
            self.pca = PCA(50, svd_solver="randomized")
            x = self.pca.fit_transform(pair_data)
        else:
            self.pca = None
        
        self.original_data = [x[treatment_data == t] for t in range(num_treatments)]
        self.ball_trees = [BallTree(x[treatment_data == t]) for t in range(num_treatments)]
        self.treatment_ids = [train_ids[treatment_data == t] for t in range(num_treatments)]
    
    def get_closest_in_propensity_lists(self, x, t, k):
        max_k = self.ball_trees[t].data.shape[0]
        adjusted_k = min(k, max_k)
        if self.pca is None:
            distance, indices = self.ball_trees[t].query(x.reshape(1, -1), k=adjusted_k)
        else:
            distance, indices = self.ball_trees[t].query(self.pca.transform(x.reshape(1, -1)), k=adjusted_k)

        idx = np.random.randint(0, len(indices))
        idx = indices[0][idx]

        chosen_sample, chosen_id = self.original_data[t][idx], self.treatment_ids[t][idx]
        return chosen_sample, chosen_id
    
    def enhance_batch_with_propensity_matches(self, x, treatment, y, inner_x, inner_t, inner_y, num_treatments, num_randomised_neighbours=6):
        all_matches = []
        for treatment_idx in range(num_treatments):
            this_treatment_indices = np.where(inner_t == treatment_idx)[0]
            matches = list(map(lambda t:
                          list(map(lambda idx: self.get_closest_in_propensity_lists(inner_x[idx], t,
                                                                               k=num_randomised_neighbours),
                              this_treatment_indices)),
                          [t_idx for t_idx in range(num_treatments) if t_idx != treatment_idx]))
            all_matches += reduce(lambda x, y: x + y, matches)


        match_ids = list(map(lambda x: x[1], all_matches))
        all_matches = np.array(list(map(lambda x: x[0], all_matches)))

        match_input_data = all_matches

        match_batch_y = y[match_ids]
        match_treatment_data = treatment[match_ids]

        return np.concatenate([inner_x, match_input_data], axis=0),\
               np.concatenate([inner_t, match_treatment_data], axis=0),\
               np.concatenate([inner_y, match_batch_y], axis=0)
    