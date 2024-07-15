import os
import numpy as np
import pandas as pd

####################################################################################

from tqdm import tqdm #Progress Bar

def load_dataframe(path):
    
    dfs = []
    for chunk in tqdm(pd.read_csv(path, header=0, chunksize=2000000), desc='Loading dataframe'):
        # chunk = chunk[chunk['subsample_ID'].isin(mask)]
        dfs.append(chunk)

    df = pd.concat(dfs).reset_index(drop=True)
    
    #df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    return df

def load_features(df, paths):
    
    #in the future, verify:
    #df_initial.shape[0] == features_py.shape[0] == features_R.shape[0]
    
    links_mask = df['link_ID']
    
    for path in paths:
        
        features_dfs = []
        
        for chunk in tqdm(pd.read_csv(path, header=0, chunksize=20000), desc='Loading features ('+path.split(os.sep)[-1]+')'):
        
            chunk = chunk[chunk['link_ID'].isin(links_mask)]
            features_dfs.append(chunk)
    
        features_df = pd.concat(features_dfs).reset_index(drop=True)
    
        features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        df = pd.merge(df, features_df, how="left", left_on='link_ID', right_on='link_ID')
    
    return df

def load_data(path_meta, path_subsample, paths_features=None, path_traits=None, biovars_path=None, reps=-1, limit=-1):
    
    # Import metadata
    meta = pd.read_csv(path_meta, header=0)
    
    # Import subsamples 
    df = load_dataframe(path = path_subsample)
    
    # Insert relevant columns from metadata file
    df = pd.merge(df, meta[['name', 'community', 'fraction', 'repetition', 'subsample_ID']], how="left", left_on='subsample_ID', right_on='subsample_ID') #'name' makes more sense for reps+multilayer (instead of 'subsample_ID')
    
    # Import features # TODO: split files into levels - network, node, link
    if paths_features:
        df = load_features(df, paths = paths_features)
    
    return meta, df