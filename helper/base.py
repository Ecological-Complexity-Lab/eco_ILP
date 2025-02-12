import os
import yaml
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

def get_local_folder():
    return os.path.dirname(os.path.realpath(__file__))

def pandas2bigraph(edgelist_df, bipartite_0 = 'lower_level', bipartite_1 = 'higher_level', weight = 'weight'):
    '''
    Converts an edgelist to networkx bipartite graph.

    Parameters
    ----------
    edgelist_df : pd.DataFrame
        edgelist.
    bipartite_0 : str, optional
        name of the column representing the first bipartite set. The default is 'lower_level'.
    bipartite_1 : str, optional
        name of the column representing the second bipartite set. The default is 'higher_level'.
    weight : str, optional
        name of the column representing the edge weight. The default is 'weight'.

    Returns
    -------
    B : networkx.Graph
        networkx bipartite graph.
    '''
    B = nx.Graph()

    # Add nodes from the first bipartite set (lower_level) with name attribute
    B.add_nodes_from(
        [(row[bipartite_0], {'bipartite': 0, 'name': row[bipartite_0]}) for _, row in edgelist_df.iterrows()]
    )
    
    # Add nodes from the second bipartite set (higher_level) with name attribute
    B.add_nodes_from(
        [(row[bipartite_1], {'bipartite': 1, 'name': row[bipartite_1]}) for _, row in edgelist_df.iterrows()]
    )

    # Filter edges with weight > 0 and add them to the graph
    edges = edgelist_df[edgelist_df[weight] > 0]
    B.add_edges_from([
        (row[bipartite_0], row[bipartite_1], {'weight': row[weight]}) for _, row in edges.iterrows()
    ])

    return B



def plot_bipartite(B):
    
    bottom_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]
    fig, ax = plt.subplots()
    # fig.set_size_inches(17, 10)
    fig.set_size_inches(17, 70)
    #nx.draw(B, nx.bipartite_layout(B, bottom_nodes, align='horizontal'), with_labels=True)
    nx.draw(B, nx.bipartite_layout(B, bottom_nodes, align='vertical'), with_labels=True)
    plt.show()

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

def load_features(df, paths, filter_features=[]):
    
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

        if len(filter_features) > 0:
            features_df = features_df[['link_ID']+filter_features]
        
        df = pd.merge(df, features_df, how="left", left_on='link_ID', right_on='link_ID')
    
    return df

def load_data(path_meta, path_subsample, paths_features=None, path_traits=None, biovars_path=None, filter_features =[], reps=-1, limit=-1):
    
    # Import metadata
    meta = pd.read_csv(path_meta, header=0)
    
    # Import subsamples 
    df = load_dataframe(path = path_subsample)
    
    # Insert relevant columns from metadata file
    df = pd.merge(df, meta[['name', 'community', 'fraction', 'repetition', 'subsample_ID']], how="left", left_on='subsample_ID', right_on='subsample_ID') #'name' makes more sense for reps+multilayer (instead of 'subsample_ID')
    
    # Import features # TODO: split files into levels - network, node, link
    if paths_features:
        df = load_features(df, paths = paths_features, filter_features=filter_features)
    
    return meta, df