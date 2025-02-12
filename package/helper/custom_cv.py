import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import BaseCrossValidator, GroupShuffleSplit, GroupKFold, StratifiedGroupKFold#, PredefinedSplit

class BaseGroupCV(BaseCrossValidator):

    def __init__(self, group_by='name', stratify_by=None, fractions_col = 'fraction', undersample_ratio=None, fractions_train = None, fractions_test = None, groups_train=None, groups_test=None, drop_isolates=False, keep_fractions_train=False, keep_fractions_test=False, trueLinks_id=None, drop_existing_links=False, random_state=0):
        
        self.group_by = group_by
        self.stratify_by = stratify_by
        self.undersample_ratio = undersample_ratio
        self.drop_isolates = drop_isolates
        self.drop_existing_links = drop_existing_links
        
        self.fractions_train = fractions_train
        self.fractions_test = fractions_test
        self.fractions_col = fractions_col

        self.groups_train = groups_train
        self.groups_test = groups_test

        self.keep_fractions_train = keep_fractions_train
        self.keep_fractions_test = keep_fractions_test

        self.trueLinks_id = trueLinks_id
        self.train_link_id = {}
        self.test_link_id = {}

        self.old_indices_mapping = None # save old index for later, as scikit-learn's KFold iterators reset the indices for some reason
        self.random_state = random_state

    def _preprocess(self, X, y=None):
        
        X = X.copy()
        
        # Split to X and y
        if y is None:
            X, y = X.iloc[:,:-1], X.iloc[:,-1]
        else:
            y = y.copy()
            
        # # Reset index & mapping old index and new one
        # indices_df = X.reset_index(drop=False).rename(columns={'index':'old_index'}).reset_index(drop=False)[['old_index', 'index']]
        # self.old_indices_mapping = dict(zip(indices_df['index'], indices_df['old_index']))

        # X = X.reset_index(drop=True)
        # y = y.reset_index(drop=True)
        
        # Deal with existing-link instances
        if self.drop_existing_links:
            if self.trueLinks_id is None:
                self.trueLinks_id = X[y == 1]['link_ID'] # Keep original existing links instances for later
            # y[y == -1] = 1 # Relabel subsampled links as existing-links

        # Get groups
        groups = np.array(X[self.group_by].values) # using name -> all reps & fracs & layers of same networks will be in the same groups

        return X, y, groups
    
    def _yield_indices(self, X, y, train_idx, test_idx):
        '''
        A function which further proccess the indices of the train and test sets, and yields the indices of the train and test sets.
        '''
        
        # Spliting the data to test set and train set
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]
        
        # # Retrieve the old indices
        # X_train.index = X_train.index.map(self.old_indices_mapping)
        # X_test.index = X_test.index.map(self.old_indices_mapping)
        # y_train.index = y_train.index.map(self.old_indices_mapping)
        # y_test.index = y_test.index.map(self.old_indices_mapping)

        # Keep only desired fraction at each dataset
        if self.keep_fractions_train == False:
            mask = X_train[self.fractions_col].isin(self.fractions_train)
            X_train = X_train[mask]
            y_train = y_train[mask]

        if self.keep_fractions_test == False:
            mask = X_test[self.fractions_col].isin(self.fractions_test)
            X_test = X_test[mask]
            y_test = y_test[mask]

        # Drop isolates
        if self.drop_isolates:
            isolate_mask = (X_train['isolate_LL'] != 1) | (X_train['isolate_HL'] != 1)
            X_train = X_train[isolate_mask]
            y_train = y_train[isolate_mask]
        
        # Undersample
        if self.undersample_ratio is not None:
            X_train, y_train = undersample(X_train, y_train, self.undersample_ratio, group_by='subsample_ID')

        if self.drop_existing_links:
            # Drop existing-links in test data | they should not get evaluated (unless test_set = 'true-network')
            # if (self.fractions_test != [1]):
            bool_remove = X_test['link_ID'].isin(self.trueLinks_id)
            X_test = X_test[~bool_remove]
            y_test = y_test[~bool_remove]

        # Save link_ids
        self._save_link_ids(X_train, X_test)
        
        return X_train.index, X_test.index
    
    def _save_link_ids(self, X_train, X_test):
        i = max(list(self.train_link_id.keys()), default=-1)+1
        self.train_link_id[i], self.test_link_id[i] = X_train['link_ID'], X_test['link_ID']

    def get_link_ids(self, i=None, as_dict=False):
        if i is None:
            if as_dict:
                return dict(self.train_link_id), dict(self.test_link_id)
            return self.train_link_id, self.test_link_id
        else:
            return self.train_link_id[i], self.test_link_id[i]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class CustomGroupCV(BaseGroupCV):

    def __init__(self, n_splits=1, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
    
    def split(self, X, y=None, groups=None):
        
        X, y, groups = self._preprocess(X, y)

        # Identify shared and unique groups
        shared_groups = set(self.groups_train).intersection(set(self.groups_test))
        train_only_groups = set(self.groups_train) - shared_groups
        test_only_groups = set(self.groups_test) - shared_groups

        # Prepare masks
        shared_mask = X[self.stratify_by].isin(shared_groups)
        train_only_mask = X[self.stratify_by].isin(train_only_groups)
        test_only_mask = X[self.stratify_by].isin(test_only_groups)

        # print(f'Shared groups: {shared_groups}')
        # print(f'Train-only groups: {train_only_groups}')
        # print(f'Test-only groups: {test_only_groups}')

        # Handle shared groups
        if shared_groups:
            X_shared = X[shared_mask]
            y_shared = y[shared_mask]
            groups_shared = groups[shared_mask]

            split_iter = self._split_groups(X_shared, y_shared, groups_shared)

            # Pre-split test-only groups to ensure each group appears only once
            test_only_split = self._split_groups(X[test_only_mask], y[test_only_mask], groups[test_only_mask])

            for shared_train_idx, shared_test_idx in split_iter:
                if train_only_groups:  # Only split train-only groups if they exist
                    train_only_split = self._split_groups(X[train_only_mask], y[train_only_mask], groups[train_only_mask])
                    train_only_train_idx, _ = next(train_only_split)
                    train_idx = np.concatenate([
                        X[train_only_mask].iloc[train_only_train_idx].index,
                        X_shared.iloc[shared_train_idx].index
                    ])
                else:
                    train_idx = X_shared.iloc[shared_train_idx].index

                if test_only_groups:  # Only include test-only groups if they exist
                    _, test_fold_idx = next(test_only_split)
                    test_idx = np.concatenate([
                        X[test_only_mask].iloc[test_fold_idx].index,
                        X_shared.iloc[shared_test_idx].index
                    ])
                else:
                    test_idx = X_shared.iloc[shared_test_idx].index

                yield self._yield_indices(X, y, train_idx, test_idx)

        else:
            # No shared groups, split train and test separately
            train_split_iter = self._split_groups(X[train_only_mask], y[train_only_mask], groups[train_only_mask])
            test_split_iter = self._split_groups(X[test_only_mask], y[test_only_mask], groups[test_only_mask])

            for (train_train_idx, _), (_, test_fold_idx) in zip(train_split_iter, test_split_iter):
                train_idx = X[train_only_mask].iloc[train_train_idx].index
                test_idx = X[test_only_mask].iloc[test_fold_idx].index
                yield self._yield_indices(X, y, train_idx, test_idx)


    
    def _split_groups(self, X, y, groups):
        if self.stratify_by:
            splitter = StratifiedGroupKFold(n_splits=self.n_splits)
            return splitter.split(X, X[self.stratify_by], groups)
        else:
            splitter = GroupKFold(n_splits=self.n_splits)
            return splitter.split(X, groups=groups)
        
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def get_link_ids(self, i=None):
        if i is None:
            return self.train_link_id, self.test_link_id
        else:
            return self.train_link_id[i], self.test_link_id[i]

class CustomGroupSplit(BaseGroupCV):

    def __init__(self, train_size=0.7, predefined_split=None, **kwargs):
        super().__init__(**kwargs)
        self.train_size = train_size
        self.predefined_split = predefined_split

    def split(self, X, y=None):
        # TODO: this function can be much more efficient by not using the entire dataframe, but only the necessary columns

        X, y, groups = self._preprocess(X, y)
                
        if self.predefined_split != None:
            train_idx = X[X[self.group_by].isin(self.predefined_split['train'])].index            
            test_idx = X[X[self.group_by].isin(self.predefined_split['test'])].index
            yield self._yield_indices(X, y, train_idx, test_idx)

        elif self.stratify_by is None:
            
            # Generate groups
            gss = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state) # train_size = proportion of the groups

            for train_idx, test_idx in gss.split(X, y, groups):
                yield self._yield_indices(X, y, train_idx, test_idx)
                break # only one split is needed

        else:
            #TODO: consider using StratifiedGroupKFold, as in CustomGroupCV

            # Set variable for convenience
            strat_col = self.stratify_by

            # Get unique values
            strat_uniques = list(X[strat_col].unique())

            # Verify that the stratify_by column contains more than one unique value
            if len(strat_uniques)<=1:
                raise ValueError('The stratify_by column should contain more than one unique value')
            
            # Generate groups - if multiple communities are present
            train_idx = []
            test_idx = []

            # Get the intersection of the train and test sets
            groups_intersection = set(self.groups_train).intersection(set(self.groups_test))

            # Loop through the unique values of the stratify_by column
            for group in strat_uniques:
                
                group_mask = X[strat_col] == group

                if group in groups_intersection:
                    # If a group is destined for both train & test set, then split it by the chosen train/test ratio
                    gss = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state)
                    train_test_idx = next(gss.split(X[group_mask], y[group_mask], groups[group_mask]))
                    train_idx += list(X[group_mask].iloc[train_test_idx[0]]['link_ID'])
                    test_idx += list(X[group_mask].iloc[train_test_idx[1]]['link_ID'])

                elif group in self.groups_train:
                    train_idx += list(X[group_mask]['link_ID'])

                elif group in self.groups_test:
                    test_idx += list(X[group_mask]['link_ID'])

            train_idx = X[X['link_ID'].isin(train_idx)].index
            test_idx = X[X['link_ID'].isin(test_idx)].index
        
            yield self._yield_indices(X, y, train_idx, test_idx)

def undersample(X, y, undersample_ratio=1, group_by=None, weights_by=None, return_removed=False):
    
    # note: use 'weights' arg of pd.sample() with shortest path..
    #weights_by = 'shortest_path_length'
    
    minor_class = 1
    major_class = 0
    
    keep_idx = [] # list of indices to keep
    
    Xy = pd.concat([X, y], axis=1)
    
    subset_columns = ['repetition', 'class', weights_by] if weights_by is not None else ['repetition', 'class']
    
    # keep the major_class of only the first repetition
    # remove_idx = Xy[(Xy['repetition']>1) & (Xy['class']==major_class)].index
    # Xy = Xy.drop(remove_idx, axis=0)

    if group_by is not None:
        gb = Xy.groupby(group_by)

        for group in gb.groups:

            subset = gb.get_group(group)[subset_columns]
            
            # if subset['repetition'].iloc[0] > 1:
                
            #     keep_idx += subset[subset['class'] == minor_class].index.tolist()
            #     continue

            # Get the sizes of the minor and major classes
            minor_class_count = (subset['class'] == minor_class).sum()
            major_class_count = (subset['class'] == major_class).sum()

            # Determine the sample size (out of the major class dataframe)
            sample_size = min(int(minor_class_count*undersample_ratio), major_class_count) # sample size cannot be larger than class size

            # Get a major class df as a subset of subsample df
            major_class_df = subset[subset['class'] == major_class]

            # Random under-sampling, keep the indices
            keep_idx += major_class_df.sample(n = sample_size, weights = weights_by).index.tolist()
            keep_idx += subset[subset['class'] == minor_class].index.tolist() # keep also the minor class indices
    
    else: # might consider to transform to function, as the code is the same as above
        
        subset = Xy[subset_columns] # faster than using whole df
        
        # Get the sizes of the minor and major classes
        minor_class_count = (subset['class'] == minor_class).sum()
        major_class_count = (subset['class'] == major_class).sum()
        
        # Determine the sample size (out of the major class dataframe)
        sumple_size = min(int(minor_class_count*undersample_ratio), major_class_count) # sample size cannot be larger than class size
        
        # Get a major class df as a subset of subsample df
        major_class_df = subset[subset['class'] == major_class]

        # Random under-sampling, keep the indices
        keep_idx += major_class_df.sample(n = sumple_size, weights = weights_by).index.tolist()
        keep_idx += subset[subset['class'] == minor_class].index.tolist() # keep also the minor class indices
    
    # Combine
    Xy_under = Xy[Xy.index.isin(keep_idx)]
    Xy_under = shuffle(Xy_under)

    # Split
    X_under = Xy_under[X.columns]
    y_under = Xy_under.iloc[:,-1]
    
    if return_removed:
        return X_under, y_under, Xy[~Xy.index.isin(keep_idx)]
    
    else:
        return X_under, y_under
