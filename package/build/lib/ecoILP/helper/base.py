import os
import yaml
import numpy as np
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt

def get_local_folder():
    return os.path.dirname(os.path.realpath(__file__))

def load_config(path=None):
    
    with open(path) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    features_meta_list = pd.read_excel(config['path']['features_meta'], header=0, sheet_name=None)
    features_meta = pd.concat(features_meta_list).reset_index(drop=True)
    features_meta = features_meta[['feature', 'level', 'package', 'fill_values_method', 'var_type', 'drop']]

    # Convert DataFrame to dictionary with values as lists
    features_dict = {}
    for index, row in features_meta.set_index('feature').iterrows():
        features_dict[index] = row.tolist()

    config['features'] = features_dict

    ##### features:level dict #####

    lvl_idx = config['features_props']['level']
    drop_idx = config['features_props']['drop']
    features = config['features']

    features_lvl = {'node':[],
                    'link':[],
                    'network':[]}

    for feature in features:
        lvl = features[feature][lvl_idx]
        drop = features[feature][drop_idx]
        if not drop:
            features_lvl[lvl].append(feature)
                            

    config['features_lvl'] = features_lvl    

    # Node features trophic level dict

    node_trophic_df = pd.read_excel(config['path']['features_meta'], header=0, sheet_name='topology')
    node_trophic_df = node_trophic_df[node_trophic_df['level']=='node'][['feature', 'node_trophic']].dropna().set_index('feature')#.reset_index(drop=True)

    # Convert DataFrame to dictionary with values as lists
    node_trophic = {}
    for index, row in node_trophic_df.iterrows():
        node_trophic[index] = row['node_trophic']

    config['node_trophic'] = node_trophic

    return config

# def sample(network, frac, weighted):
#     '''
#     generate a single sample.

#     Parameters
#     ----------
#     network : pd.DataFrame
#         Adjency matrix.
#     frac : float
#         Fraction of the links that will be removed.
#     weighted : bool, optional
#         True if using egde weights. The default is False.

#     Returns
#     -------
#     network_sample : pd.DataFrame
#         Adjency matrix as 'network' argument, with removed links according to 'frac' argument.

#     '''
#     network_flat = network.to_numpy().flatten() # random.multinomial accept only 1d arrays
    
#     if weighted == False:

#         # Pick random obs interactions
#         obs_idx = np.random.choice(a=network_flat.nonzero()[0], size=int(sum(network_flat)*frac), replace=False) 

#         # Initiate empty matrix and fill observations with 'obs_idx'
#         network_sample = np.zeros(network_flat.shape[0])
#         network_sample[obs_idx]=1
        
#     else:
        
#         weights_sum = round(network.values.sum()*frac,0) # Desired number of weights (total)
#         network_probs = network_flat/network_flat.sum() # calc [proba]
#         network_sample = np.random.multinomial(n=weights_sum, pvals=network_probs, size=1) # Generate sample out of multinomial distribution
    
#     network_sample = network_sample.reshape(network.shape[0], network.shape[1]) # Reshape back to matrix
#     network_sample = pd.DataFrame(network_sample, index=network.index, columns=network.columns) # Change back to dataframe
    
#     return network_sample
        
    #print(nx.number_connected_components(B), nx.number_connected_components(B_sample))
    #print(nx.number_of_isolates(B), nx.number_of_isolates(B_sample))
    

####################################################################################


# components_df = pd.DataFrame() ### for debugging, delete if not used

# def create_samples(network, network_edgelist, frac_list, reps = 1, weighted = False, min_components = False):
#     '''
#     Generating subsamples.

#     Parameters
#     ----------
#     network : pd.DataFrame
#         Adjency matrix.
#     network_edgelist : pd.DataFrame
#         Edge list.
#     frac_list : list
#         list of desired fractions.
#     reps : int, optional
#         number of repetition of subsampling. The default is 1.
#     weighted : bool, optional
#         True if using egde weights. The default is False.
#     min_components : bool, optional
#         True if the algorithm should minimize the number of components of each generated sample.
    
#     Returns
#     -------
#     subsamples : dict
#         Containing all generated subsamples

#     '''
    
#     if weighted == False: 
#         network = network > 0 # convert weighted networks to binary
    
#     # Create a dictionary storing all subsamples - for each fraction
#     subsamples = {frac:{} for frac in frac_list+[1]}
    
#     B = pandas2bigraph(network_edgelist) # Convert pandas dataframe to networkx bipartite graph
        
#     for frac in frac_list:
                
#         for rep in range(1, reps+1):
            
#             if min_components:
                
#                 subsamples_dict = {} # Create empty dict containing subsampled networks
                
#                 # was range(20), but I don't want multople repetitions to be the same
#                 for i in range(2): # Generate subsamples, redo the random sampling if subsample is not connected
#                     network_sample = sample(network, frac, weighted)
#                     network_sample_edgeList = network_sample.stack().rename_axis(['lower_level','higher_level']).rename('weight').reset_index() #Create an edge list
#                     B_sample = pandas2bigraph(network_sample_edgeList)
                    
#                     subsamples_dict[nx.number_connected_components(B_sample)] = (network_sample, network_sample_edgeList)
#                     if nx.number_connected_components(B_sample) == nx.number_connected_components(B):
#                         break
                    
#                 network_sample, network_sample_edgeList = subsamples_dict[min(subsamples_dict.keys())]
                
#             else:
#                 network_sample = sample(network, frac, weighted)
#                 network_sample_edgeList = network_sample.stack().rename_axis(['lower_level','higher_level']).rename('weight').reset_index() #Create an edge list
            
#             #Set a class for each link, wether its TP, TN or Missed Obs
#             TP = (network > 0) & (network_sample > 0) #TP
#             TN = (network == 0) & (network_sample == 0) #TN
#             FN = (network > 0) & (network_sample == 0) #Missed Obs
            
#             network_sample_edgeList = pd.merge(network_sample_edgeList, TP.stack().rename_axis(['lower_level','higher_level']).rename('TP').reset_index(), how='left', on=['lower_level', 'higher_level'])
#             network_sample_edgeList = pd.merge(network_sample_edgeList, TN.stack().rename_axis(['lower_level','higher_level']).rename('TN').reset_index(), how='left', on=['lower_level', 'higher_level'])
#             network_sample_edgeList = pd.merge(network_sample_edgeList, FN.stack().rename_axis(['lower_level','higher_level']).rename('FN').reset_index(), how='left', on=['lower_level', 'higher_level'])
        
#             network_sample_edgeList.loc[network_sample_edgeList.TP == True, 'class'] = "TP"
#             network_sample_edgeList.loc[network_sample_edgeList.TN == True, 'class'] = "TN"
#             network_sample_edgeList.loc[network_sample_edgeList.FN == True, 'class'] = "FN"
            
#             network_sample_edgeList = network_sample_edgeList.drop(["TP", "TN", "FN"], axis=1)
            
#             subsamples[frac][rep] = {"Adjacency_Matrix":network_sample,
#                                      "Edge_List":network_sample_edgeList}
    
    
#     # Add also frac = 1 (Original network)
#     network_edgelist['class'] = network_edgelist.apply(lambda row: 'TP' if row['weight'] > 0 else 'TN', axis=1)
    
#     subsamples[1][1] = {"Adjacency_Matrix":network,
#                           "Edge_List":network_edgelist}
    
#     #global components_df ###
#     #components_df = pd.concat([components_df, pd.DataFrame({'network':net_id, 'connected':nx.is_connected(B), 'components':nx.number_connected_components(B), 'singletons':nx.number_of_isolates(B)}, index=[0])], axis=0) ###

#     # return subsamples

#     # --------------------------------------
#     # convert to df
#     dfs = []  # List to store dataframes with added columns

#     # Iterate over the outer and inner dictionaries
#     for fraction, reps_dict in subsamples.items():
#         for rep, net_data in reps_dict.items():
#             df = net_data['Edge_List']
#             df['fraction'] = fraction
#             df['repetition'] = rep
#             dfs.append(df)

#     # Concatenate all dataframes along axis 0
#     result_df = pd.concat(dfs, axis=0).reset_index(drop=True)

#     return result_df

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


# Load bioclimatic variables (Enviorment)
def load_env(df, path):
    
    env_df = pd.read_csv(path, header=0, index_col=0) # Reading the file
    df = pd.merge(df, env_df, how="left",left_on='name',right_on='ID') # Merging with df
        
    # Check for missing values
    missing_vars = set(df.name.unique())-(set(env_df.index.unique()))
    if len(missing_vars) > 0:
        print("Networks with missing bioclimatic variables: ", missing_vars) # Difference between the network ID's
    
    return df

def load_traits(df, type_='seed-dispersal'):
    
    # birds_traits_ = birds_traits[['original_name', 'Beak.Length_Culmen', 'Beak.Length_Nares', 'Beak.Width', 'Beak.Depth', 
    #                               'Tarsus.Length', 'Wing.Length', 'Kipps.Distance', 'Secondary1','Hand-wing.Index', 'Tail.Length']].fillna(0)
    
    # df_ = pd.merge(df[df['community']=='Plant-Seed Dispersers'], birds_traits_, how="left", left_on='higher_level',right_on='original_name')
    
    # df_ = df_.drop(df_[df_['original_name'].isna()].index, axis=0)
    
    # df_ = df_.drop(['original_name'], axis=1, errors='ignore')
    
    # df = df_.reset_index(drop=True).copy()
    
    return None

def load_data(path_meta, path_subsample, paths_features=None, path_traits=None, biovars_path=None, filter_features =[], reps=-1, limit=-1):
    
    # Import metadata
    meta = pd.read_csv(path_meta, header=0)
    
    # Import subsamples 
    df = load_dataframe(path = path_subsample)
    
    # Insert relevant columns from metadata file
    df = pd.merge(df, meta[['name', 'community', 'fraction', 'repetition', 'subsample_ID']], how="left", left_on='subsample_ID', right_on='subsample_ID') #'name' makes more sense for reps+multilayer (instead of 'subsample_ID')
    
    # Import bioclimatic variables (Enviorment)
    if biovars_path:
        df = load_env(df, path = biovars_path)
    
    # Import features # TODO: split files into levels - network, node, link
    if paths_features:
        df = load_features(df, paths = paths_features, filter_features=filter_features)
    
    # Import traits
    if path_traits:
        df = load_traits(df)
    
    return meta, df