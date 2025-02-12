import logging
import pandas as pd
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

def custom_scorer(y_true, y_pred, metric='f1', exclude_mask=None, cast_links=False):
    '''
    Custom scorer for GridSearchCV.
    Is in seperate file to avoid parallelization issues.
    '''
    
    if exclude_mask is not None:
        ignore = y_true.index.isin(exclude_mask)

        # Apply the mask
        y_true = y_true[~ignore]
        y_pred = y_pred[~ignore]

    if cast_links: # TODO: Maybe there is no need anymore
        y_true[y_true == -1] = 1
    
    if metric == 'f1':
        return f1_score(y_true, y_pred)
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    # elif metric == 'mcc':
    #     return matthews_corrcoef(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    