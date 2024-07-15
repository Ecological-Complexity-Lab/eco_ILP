# %% [markdown]
# ## Import Packages

# %%
#%% Import Packages

import os
# import pathlib
import pandas as pd

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from helper.topology_functions import extract_features

# Number of avialable cores
nslots = int(os.getenv('NSLOTS', 2)) # number of cores | default is 2
print('cores:' ,nslots, ', dir:', os.getcwd()) # should be '/gpfs0/shai/users/barryb/link-predict'


# %% [markdown]
# ## Load data

# %%
print('Loading dataframe', flush=True)
df = pd.read_csv('data/processed/networks/subsamples_edge_lists.csv', header=0)

print('Loading metadata', flush=True)
df_meta = pd.read_csv('data/processed/networks/subsamples_metadata.csv', header=0)

df['lower_level'] = df['lower_level'].astype(str) # TODO: fix this in the data processing (I thought I did but there are errors)
df['higher_level'] = df['higher_level'].astype(str)


# %% [markdown]
# ## Set configurations

# %% [markdown]
# Select features

# %%
features_list = [
    'network_size', 
    'species_ratio', 
    'interactions_count', 
    'edge_connectivity', # sometimes a bit slow
    'density', 
    'bipartite_clustering', # slow | 
    'Spectral_bipartivity', # sometimes a bit slow (?)
    'average_clustering', 
    'degree_assortativity_coefficient', 
    'global_efficiency', # sometimes a bit slow
    'local_efficiency', # sometimes a bit slow
    'connected_components', 
    'degree', 
    'latapy_clustering', 
    'node_redundancy', # sometimes slow | 
    'betweenness_centrality', # a bit slow
    'degree_centrality', 
    'closeness_centrality', 
    'average_neighbor_degree', 
    'pagerank', 
    'hits_hubs',
    'hits_authorities', 
    'isolate', 
    'preferential_attachment', 
    'shortest_path_length', 
    'shortest_paths_count', # slow | 
    'friends_measure', # slow
    'same_community_infomap', # Also returns 'flow_infomap' and 'modular_centrality # slow
]


# %%
multiprocess = True # parallel processing

# %% [markdown]
# ## Extract Network Features

# %%

if multiprocess == True:
    extract_features(df, features_list, nslots=nslots, table_path='data/processed/features/features_py.csv')
else:
    extract_features(df, features_list, nslots=1, table_path='data/processed/features/features_py.csv')

# %%
print('Done!')