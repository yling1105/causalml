"""
Filter feature selection methods for uplift modeling

- Currently only for classification problem: the outcome variable of uplift model is binary.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
    
class FilterSelect:
    """A class for feature importance methods.
    """

    def __init__(self):
        return

    def _convert_treatment(self, data, T_name_before, treatment_key):
        """
        Convert text treatment labels into 0 (control) and 1 (treated).
        """
        T_dummies = pd.get_dummies(data[T_name_before])
        T_dummies_treatment = T_dummies[treatment_key]   
        return T_dummies_treatment


    def _filter_F_one_feature(self, data, T_name, X_name, y_name): 
        """
        Conduct F-test of the interaction between treatment and one feature.
        """
        Y = data[y_name]
        X = data[[T_name, X_name]]
        X = sm.add_constant(X)
        X['{}-{}'.format(T_name, X_name)] = X[[T_name, X_name]].product(axis=1)

        model = sm.OLS(Y, X)
        result = model.fit()

        F_test = result.f_test(np.array([0, 0, 0, 1]))
        F_test_result = pd.DataFrame({
            'feature': X_name, # for the interaction, not the main effect
            'method': 'F-statistic',
            'score': F_test.fvalue[0][0], 
            'p_value': F_test.pvalue, 
            'misc': 'df_num: {}, df_denom: {}'.format(F_test.df_num, F_test.df_denom), 
        }, index=[0]).reset_index(drop=True)

        return F_test_result


    def filter_F(self, data, T_name, X_name_list, y_name):
        """
        Rank features based on the F-statistics of the interaction.
        """
        all_result = pd.DataFrame()
        for x_name_i in X_name_list: 
            one_result = self._filter_F_one_feature(data=data,
                T_name=T_name, X_name=x_name_i, y_name=y_name
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by='score', ascending=False)
        all_result['rank'] = all_result['score'].rank(ascending=False)

        return all_result


    def _filter_LR_one_feature(self, data, T_name, X_name, y_name, disp=True): 
        """
        Conduct LR (Likelihood Ratio) test of the interaction between treatment and one feature.
        """
        Y = data[y_name]
        
        # Restricted model
        X_r = data[[T_name, X_name]]
        X_r = sm.add_constant(X_r)
        model_r = sm.Logit(Y, X_r)
        result_r = model_r.fit(disp=disp)

        # Full model (with interaction)
        X_f = X_r.copy()
        X_f['{}-{}'.format(T_name, X_name)] = X_f[[T_name, X_name]].product(axis=1)
        model_f = sm.Logit(Y, X_f)
        result_f = model_f.fit(disp=disp)

        LR_stat = -2 * (result_r.llf - result_f.llf)
        LR_df = len(result_f.params) - len(result_r.params)
        LR_pvalue = 1 - stats.chi2.cdf(LR_stat, df=LR_df)

        LR_test_result = pd.DataFrame({
            'feature': X_name, # for the interaction, not the main effect
            'method': 'LRT-statistic',
            'score': LR_stat, 
            'p_value': LR_pvalue,
            'misc': 'df: {}'.format(LR_df), 
        }, index=[0]).reset_index(drop=True)

        return LR_test_result


    def filter_LR(self, data, T_name, X_name_list, y_name, disp=True):
        """
        Rank features based on the LRT-statistics of the interaction.
        """
        all_result = pd.DataFrame()
        for x_name_i in X_name_list: 
            one_result = self._filter_LR_one_feature(data=data, 
                T_name=T_name, X_name=x_name_i, y_name=y_name, disp=disp
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by='score', ascending=False)
        all_result['rank'] = all_result['score'].rank(ascending=False)

        return all_result


    # Get node summary - a function 
    def _GetNodeSummary(self, data, 
                        experiment_group_column='treatment_group_key', 
                        y_name='conversion'):
        """
        To count the conversions and get the probabilities by treatment groups.

        Parameters
        ----------
        data : DataFrame
            The DataFrame that contains all the data (in the current "node").  
            - [TODO] We may need a index vector to restrict the rows of data 
              outside the function.
            - [TODO] Currently the positions of treatment_group_key and 
              conversion in data are hard coded.  Modify the code for more
              flexibility.

        Returns
        -------
        results : dict
            Counts of conversions by treatment groups, of the form: 
            {'control': {0: 10, 1: 8}, 'treatment1': {0: 5, 1: 15}}
        nodeSummary: dict
            Probability of conversion and group size by treatment groups, of 
            the form:
            {'control': [0.490, 500], 'treatment1': [0.584, 500]}

        References
        ----------
        - results: upliftpy.UpliftTreeClassifier.group_uniqueCounts() 
        - nodeSummary: upliftpy.UpliftTreeClassifier.tree_node_summary()
        """

        # Note: results and nodeSummary are both dict with treatment_group_key
        # as the key.  So we can compute the treatment effect and/or 
        # divergence easily.

        # Counts of conversions by treatment group
        results_series = data.groupby([experiment_group_column, y_name]).size()
        
        treatment_group_keys = results_series.index.levels[0].tolist()
        y_name_keys = results_series.index.levels[1].tolist()

        results = {}
        for ti in treatment_group_keys: 
            results.update({ti: {}}) 
            # results[ti] = {} # alternatively
            for ci in y_name_keys:
                results[ti].update({ci: results_series[ti, ci]}) 
                # results[ti][ci] = results_series[ti, ci] # alternatively

        # Probability of conversion and group size by treatment group
        nodeSummary = {}
        for treatment_group_key in results: 
            n_1 = results[treatment_group_key][1]
            n_total = (results[treatment_group_key][1] 
                       + results[treatment_group_key][0])
            y_mean = 1.0 * n_1 / n_total
            nodeSummary[treatment_group_key] = [y_mean, n_total]
        # [TODO] More complicated situations in tree_node_summary()
        
        return results, nodeSummary 

    # Divergence-related functions, from upliftpy
    def _kl_divergence(self, pk, qk):
        """
        Calculate KL Divergence for binary classification.

        Parameters
        ----------
        pk, qk : float
        """
        # Formula
        # S = sum(np.array(pk) * np.log(np.array(pk) / np.array(qk)))
        if qk < 0.1**6:
            qk = 0.1**6
        elif qk > 1 - 0.1**6:
            qk = 1 - 0.1**6
        S = pk * np.log(pk / qk) + (1-pk) * np.log((1-pk) / (1-qk))
        return S

    def _evaluate_KL(self, nodeSummary, control_group='control'):
        """
        Calculate the multi-treatment unconditional D (one node)
        with KL Divergence as split Evaluation function.

        Parameters
        ----------
        control_group : str, optional (default='control')

        Notes
        -----
        The function works for more than one non-control treatment groups.
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += self._kl_divergence(nodeSummary[treatment_group][0], pc)
        return d_res

    def _evaluate_ED(self, nodeSummary, control_group='control'):
        """
        Calculate the multi-treatment unconditional D (one node)
        with Euclidean Distance as split Evaluation function.
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += 2 * (nodeSummary[treatment_group][0] - pc)**2
        return d_res

    def _evaluate_Chi(self, nodeSummary, control_group='control'):
        """
        Calculate the multi-treatment unconditional D (one node)
        with Chi-Square as split Evaluation function.
        """
        if control_group not in nodeSummary:
            return 0
        pc = nodeSummary[control_group][0]
        d_res = 0
        for treatment_group in nodeSummary:
            if treatment_group != control_group:
                d_res += (
                    (nodeSummary[treatment_group][0] - pc)**2 / max(0.1**6, pc) 
                    + (nodeSummary[treatment_group][0] - pc)**2 / max(0.1**6, 1-pc)
                )
        return d_res


    def _filter_D_one_feature(self, data, X_name, y_name, 
                              n_bins=10, method='KL', control_group='control',
                              experiment_group_column='treatment_group_key'):
        """
        Calculate the chosen divergence measure for one feature.
        """
        # [TODO] Application to categorical features

        if method == 'KL':
            evaluationFunction = self._evaluate_KL
        elif method == 'ED':
            evaluationFunction = self._evaluate_ED
        elif method == 'Chi':
            evaluationFunction = self._evaluate_Chi

        totalSize = len(data.index)
        x_bin = pd.qcut(data[X_name].values, n_bins, labels=False, 
                        duplicates='drop')
        d_children = 0
        for i_bin in range(x_bin.max() + 1): # range(n_bins):
            nodeSummary = self._GetNodeSummary(
                data=data.loc[x_bin == i_bin], 
                experiment_group_column=experiment_group_column, y_name=y_name
            )[1]
            nodeScore = evaluationFunction(nodeSummary, 
                                           control_group=control_group)
            nodeSize = sum([x[1] for x in list(nodeSummary.values())])
            d_children += nodeScore * nodeSize / totalSize

        parentNodeSummary = self._GetNodeSummary(
            data=data, experiment_group_column=experiment_group_column, y_name=y_name
        )[1]
        d_parent = evaluationFunction(parentNodeSummary, 
                                      control_group=control_group)
            
        d_res = d_children - d_parent
        
        D_result = pd.DataFrame({
            'feature': X_name, 
            'method': method,
            'score': d_res, 
            'p_value': None,
            'misc': 'number_of_bins: {}'.format(min(n_bins, x_bin.max()+1)),# format(n_bins),
        }, index=[0]).reset_index(drop=True)

        return(D_result)

    def filter_D(self, data, X_name_list, y_name, 
                 n_bins=10, method='KL', control_group='control',
                 experiment_group_column='treatment_group_key'):
        """
        Rank features based on the chosen divergence measure.
        """
        
        all_result = pd.DataFrame()

        for x_name_i in X_name_list: 
            one_result = self._filter_D_one_feature(
                data=data, X_name=x_name_i, y_name=y_name,
                n_bins=n_bins, method=method, control_group=control_group,
                experiment_group_column=experiment_group_column, 
            )
            all_result = pd.concat([all_result, one_result])

        all_result = all_result.sort_values(by='score', ascending=False)
        all_result['rank'] = all_result['score'].rank(ascending=False)

        return all_result

    def get_importance(self, data, X_name_list, y_name, method, 
                      experiment_group_column='treatment_group_key',
                      control_group = 'control', 
                      treatment_group = 'treatment',
                      n_bins=5, 
                      ):
        """
        Rank features based on the chosen statistic of the interaction.

        Args:
            data (pd.Dataframe): DataFrame containing outcome, features, and experiment group
            method (string): {'F', 'LR', 'KL', 'ED', 'Chi'}
                The feature selection method to be used to rank the features.  
                'F' for F-test 
                'LR' for likelihood ratio test
                'KL', 'ED', 'Chi' for bin-based uplift filter methods
            experiment_group_column (string): the experiment column name in the DataFrame, which contains the treatment and control assignment label
            control_group (string): name for control group, value in the experiment group column
            treatment_group (string): name for treatment group, value in the experiment group column
            n_bins (int, optional): number of bins to be used for bin-based uplift filter methods
        
        Returns:
            (pd.DataFrame): a data frame with following columns: ['method', 'feature', 'rank', 'score', 'p_value', 'misc']
        """
        
        if method == 'F':
            data = data[data[experiment_group_column].isin([control_group, treatment_group])]
            data['T_name'] = 0
            data.loc[data[experiment_group_column]==treatment_group,'T_name'] = 1
            all_result = self.filter_F(data=data, 
                T_name='T_name', X_name_list=X_name_list, y_name=y_name
            )
        elif method == 'LR':
            data = data[data[experiment_group_column].isin([control_group, treatment_group])]
            data['T_name'] = 0
            data.loc[data[experiment_group_column]==treatment_group,'T_name'] = 1
            all_result = self.filter_LR(data=data, disp=True,
                T_name='T_name', X_name_list=X_name_list, y_name=y_name
            )
        else:
            all_result = self.filter_D(data=data, method=method,
                X_name_list=X_name_list, y_name=y_name, 
                n_bins=n_bins, control_group=control_group,
                experiment_group_column=experiment_group_column, 
            )
        
        all_result['method'] = method + ' filter'
        return all_result[['method', 'feature', 'rank', 'score', 'p_value', 'misc']]