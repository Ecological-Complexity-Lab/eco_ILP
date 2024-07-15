import logging
import pandas as pd
from sklearn.metrics import f1_score #, balanced_accuracy_score, matthews_corrcoef

def custom_scorer(y_true, y_pred, exclude_mask=None, cast_links=False):
    '''
    Custom scorer for GridSearchCV.
    Is in seperate file to avoid parallelization issues.
    '''

    # Set up the logger
    # logging.basicConfig(filename='custom_scorer.log', level=logging.INFO)
    # logger = logging.getLogger(__name__)

    # Log the original indices of y_true
    # logger.info(f"Original indices of y_true: {y_true.index}")
    
    if exclude_mask is not None:
        ignore = y_true.index.isin(exclude_mask)

        # Apply the mask
        y_true = y_true[~ignore]
        y_pred = y_pred[~ignore]

    if cast_links: # TODO: Maybe there is no need anymore
        y_true[y_true == -1] = 1
    
    # Log the indices of y_true after applying the mask
    # logger.info(f"Indices of y_true after applying the mask: {y_true.index}")
    
    # return balanced_accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred)
    
    