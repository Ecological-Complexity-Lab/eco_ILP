# Import Packages

import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools


from infomap import Infomap
import networkx as nx
from networkx.algorithms import bipartite

import multiprocessing
from tqdm import tqdm # Progress Bar

from helper.base import pandas2bigraph
# from helper.motifs_functions import getMotifs, plot_motif

# Functions

def dict2df(network, features_dict, top_nodes, bottom_nodes):
    '''
    Convert a dictionary, which contains all computed features for a given subsample, to a dataframe - formatted for exporting to csv/sql database.

    Parameters
    ----------
    network : pd.DataFrame
        An edgelist format of a sampled network.
    features_dict : dict
        A dictionary containing the features and values, nested by the scope of the feaute(network-level node-level..).
    top_nodes : list
        The set of top nodes.
    bottom_nodes : list
        The set of bottom nodes.

    Returns
    -------
    network : pd.DataFrame
        An edgelist format of a sampled network, including the features as columns.

    '''
    
    drop_columns = list(network.columns) # drop (later) all columns which are not 'link_ID'
    drop_columns.remove('link_ID')
    
    # Add network-level features to edge list
    if 'network_level' in features_dict:
        network_level = features_dict["network_level"]
        network = network.assign(**network_level)
    
    # Add nodes-level features to edge list
    if 'node_level' in features_dict:
        node_level = pd.concat([pd.DataFrame({key: val}) for key, val in features_dict["node_level"].items()], axis=1)
        network = network.merge(node_level.add_suffix('_LL'), left_on='lower_level', right_index=True, how = 'left')
        network = network.merge(node_level.add_suffix('_HL'), left_on='higher_level', right_index=True, how = 'left')
    
    # Add link-level features to edge list
    if 'link_level' in features_dict:
        dfs = [pd.DataFrame(val, columns=['higher_level', 'lower_level', key]) for key, val in features_dict["link_level"].copy().items()]
        
        for df in dfs:
            
            # Drop rows where both species are from the same trophic level
            same_trophic_level = df[(df['higher_level'].isin(top_nodes) & df['lower_level'].isin(top_nodes)) | (df['higher_level'].isin(bottom_nodes) & df['lower_level'].isin(bottom_nodes))] 
            df.drop(same_trophic_level.index, inplace = True)
            
            # Switch between 'higher_level' and 'lower_level' when the species are in the wrong order
            to_switch = ~df['higher_level'].isin(bottom_nodes)
            df.loc[to_switch, ['higher_level', 'lower_level']] = (df.loc[to_switch, ['lower_level', 'higher_level']].values)
            
            # Drop duplicates
            df.drop_duplicates(inplace = True)
            
            # Set multiindex for easier concat
            df.set_index(['lower_level', 'higher_level'], inplace = True)
        
        link_level = pd.concat(dfs, axis=1)
        
        network = network.merge(link_level, left_on=['higher_level', 'lower_level'], right_index=True, how = 'left')
    
    # # Add communities to edge list | not very efficient, i should improve this part | delete
    # if 'communities' in features_dict:
    #     for algorithm, communities in features_dict["communities"].items():
    #         network[algorithm] = False
    #         for index, row in network.iterrows():
    #             for community in communities:
    #                 if row['higher_level'] in community and row['lower_level'] in community:
    #                     network.loc[index, algorithm] = True
    
    
    # Add fitted models to edge list
    if 'fitted_models' in features_dict:
        network = network.merge(features_dict["fitted_models"]["SBM_probs"], how='left', on=['lower_level', 'higher_level'])
        
    # Add fitted models to edge list
    # if 'Node2Vec' in features_dict:
    #     network = network.merge(features_dict["Node2Vec"], how='left', left_on='lower_level', right_index=True)
    #     network = network.merge(features_dict["Node2Vec"], how='left', left_on='higher_level', right_index=True)
    
    # drop columns which are not new features
    network = network.drop(drop_columns, axis = 1)
    
    return network

####################################################################################

# Friends measure
def friends_measure(G, u, v):
    '''
    Compute friends measure for a pair of nodes

    Parameters
    ----------
    G : networkx.Graph
        network.
    u : ?
        node.
    v : ?
        node.

    Returns
    -------
    score : pd.DataFrame
        An edgelist format of a sampled network, including the features as columns.

    '''
    #https://github.com/sweety-dhale/Data_Mining/tree/40d16c1306423f05e9f6eaa16f2af046d9309760
    #https://github.com/mcaballeroguillen/tesis/blob/2981334c1ab7ec5963e6d03a6290d94cc907ab5c/SimRank/networkx_addon/similarity/katz.py
    score = 0
    for x in G[u]:
        for y in G[v]:
            if (x == y) or G.has_edge(x,y):
                score = score + 1
    return score

####################################################################################

# Katz measure
def katz_measure(G, beta=0.05, non_edges=[], bottom_nodes=None, top_nodes=None, node_u=None, node_v=None):
    # Get the adjacency matrix of the graph
    A = nx.adjacency_matrix(G)
    
    # Calculate the identity matrix
    I = np.eye(G.number_of_nodes())
    
    # Compute the Katz Index
    # Katz(u, v) = (I - beta * A)^-1 - I
    try: 
        katz_matrix = np.linalg.pinv(I - beta * A) - I # http://www.cs.utexas.edu/~yzhang/papers/osn-imc09.pdf
        # Katz_matrix = np.linalg.inv(I - beta * A) - I #using inv #https://stackoverflow.com/questions/49357417/why-is-numpy-linalg-pinv-preferred-over-numpy-linalg-inv-for-creating-invers
    except np.linalg.LinAlgError:
        print("SVD did not converge. Adjusting beta...")
        
        # Adjust beta to a smaller value and retry
        beta = beta * 0.1  # or any other value
        try:
            katz_matrix = np.linalg.pinv(I - beta * A) - I
        except np.linalg.LinAlgError:
            print("Adjustment failed. Returning default matrix...")
            katz_matrix = -1 * np.ones(I.shape)

    if node_u and node_v:
        # Get the index of the nodes in the adjacency matrix
        node_u_idx = list(G.nodes()).index(node_u)
        node_v_idx = list(G.nodes()).index(node_v)
        katz_value = katz_matrix[node_u_idx, node_v_idx]
        return katz_value
    
    elif bottom_nodes and top_nodes:
        nodes = list(G.nodes())
        katz_tuples = []

        for i in range(len(nodes)):
            for j in range(len(nodes)):  # we only need to consider pairs once
                if (nodes[i], nodes[j]) in non_edges: # filter the edges
                    if (nodes[i] in bottom_nodes) and (nodes[j] in top_nodes): # verify node levels
                        katz_tuples.append((nodes[i], nodes[j], katz_matrix[i, j]))
        return katz_tuples
    else:
        raise Exception('error')

####################################################################################

def bigraph_relable_to_integer(B, return_sep=False):

    bottom_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]
    top_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 1]

    mapping = {name : i+1 for i, name in enumerate(top_nodes+bottom_nodes)}

    B = nx.relabel_nodes(B, mapping)

    if return_sep:
        return B, len(top_nodes)+1

    return B

def infomap_modules(B, return_df=False):
    
    B_numerical_id, bipartite_start_id = bigraph_relable_to_integer(B, return_sep=True) # Relable nodes to integers
    im = Infomap(two_level=True, silent=True) # Create Infomap object
    im.add_networkx_graph(B_numerical_id) # Add networkx graph to Infomap object
    im.bipartite_start_id = bipartite_start_id # Set the start id for bipartite nodes
    im.run() # Run the Infomap search algorithm to find optimal modules

    modules_df = im.get_dataframe(columns=["node_id", "module_id", "flow"]) # Get the modules as a pandas dataframe
    modules_df['node_name'] = [B_numerical_id.nodes[node_id]['name'] for node_id in modules_df['node_id']]  # Add the node names as a column
    modules_df['bipartite'] = [B_numerical_id.nodes[node_id]['bipartite'] for node_id in modules_df['node_id']]

    if return_df:
        return modules_df
    
    # Filtering nodes based on bipartite values
    bipartite_0 = modules_df[modules_df['bipartite'] == 0]
    bipartite_1 = modules_df[modules_df['bipartite'] == 1]

    # Creating tuples for each combination and checking if they are in the same community (same module_id)
    same_community = []
    for _, row_0 in bipartite_0.iterrows():
        for _, row_1 in bipartite_1.iterrows():
            in_same_community = int(row_0['module_id'] == row_1['module_id'])
            same_community.append((row_0['node_name'], row_1['node_name'], in_same_community))

    # Creating the 'flow' dictionary
    flow_dict = dict(zip(modules_df['node_name'], modules_df['flow']))

    return same_community, flow_dict

####################################################################################

def getBridges(B, bottom_nodes, non_edges):
    
    # get current bridges out of existing links
    bridges = list(nx.bridges(B)) 
    bridges = [(x, y, 1) if x in bottom_nodes else (y, x, 1) for x, y in bridges] # fix mixing of species order made by nx.bridges() + add a value

    B_copy = B.copy() # don't modify original graph

    # get possible bridges out of non-existing links
    for bottom, top in non_edges:
        B_copy.add_edge(bottom, top)
        bridges_ = list(nx.bridges(B_copy))

        if bridges_.count((bottom, top)) > 0 or bridges_.count((top, bottom)) > 0:
            bridges.append((bottom, top, 1))

        B_copy.remove_edge(bottom, top)
    
    return bridges


####################################################################################
    
def getNetFeatures(edgelist_df, features_list=[]):
    '''
    """Extract features from a single network"""
    Compute all topological features for a given network, return a ready-to-export dataframe.
    
    Parameters
    ----------
    edgelist_df : pd.DataFrame
        DESCRIPTION.
    
    Returns
    -------
    features_dict : pd.DataFrame
        DESCRIPTION.
    
    '''
    features_dict = {}
    
    B = pandas2bigraph(edgelist_df)
    
    edgelist_minimal = edgelist_df[edgelist_df['weight'] != 0][['lower_level', 'higher_level']]
        
    ## get top and bottom nodes    
    bottom_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]
    top_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 1]
    
    # Get all possible edges
    possible_edges = list(itertools.product(bottom_nodes, top_nodes))
    
    # Get existing edges
    edges = list(zip(edgelist_minimal.lower_level, edgelist_minimal.higher_level))
    
    # Get non-existing edges
    non_edges = list(set(possible_edges).difference(set(edges)))
    
    # Verify length of above lists
    if len(possible_edges) != (len(edges) + len(non_edges)):
        raise Exception("possible_edges != edges + non_edges") 
    

    # Create empty dictionaries for each scope
    features_dict["network_level"] = {} # Create "network_level" key
    features_dict["node_level"] = {} # Create "node_level" key
    features_dict["link_level"] = {} # Create "link_level" key

    ############################################
    ############### Network Level ##############
    ############################################

    # Network Size
    if 'size' in features_list:
        features_dict["network_level"]["size"] = len(top_nodes) + len(bottom_nodes)
    
    # Network Ratio
    if 'species_ratio' in features_list:
        features_dict["network_level"]["species_ratio"] = len(top_nodes)/len(bottom_nodes)
    
    # Interactions Count
    if 'interactions_count' in features_list:
        features_dict["network_level"]["interactions_count"] = sum(edgelist_df['weight'] > 0)
    
    # Edge Connectivity
    if 'edge_connectivity' in features_list:
        features_dict["network_level"]["edge_connectivity"] = nx.edge_connectivity(B)
    
    # Density
    if 'density' in features_list:
        features_dict["network_level"]["density"] = bipartite.density(B, top_nodes)
    
    # Bipartite Clustering (Robins and Alexander)
    if 'bipartite_clustering' in features_list:
        features_dict["network_level"]["bipartite_clustering"] = bipartite.robins_alexander_clustering(B)
    
    if 'Spectral_bipartivity' in features_list:
        features_dict["network_level"]["Spectral_bipartivity"] = bipartite.spectral_bipartivity(B)
    
    if 'average_clustering' in features_list:
        features_dict["network_level"]["average_clustering"] = bipartite.average_clustering(B)
    
    # Assortativity
    if 'degree_assortativity_coefficient' in features_list:
        features_dict["network_level"]["degree_assortativity_coefficient"] = nx.degree_assortativity_coefficient(B)
    
    # Efficiency
    if 'global_efficiency' in features_list:
        features_dict["network_level"]["global_efficiency"] = nx.global_efficiency(B)

    if 'local_efficiency' in features_list:
        features_dict["network_level"]["local_efficiency"] = nx.local_efficiency(B)
    
    # Connected components
    if 'connected_components' in features_list:
        features_dict["network_level"]["connected_components"] = nx.number_connected_components(B)
    
    ##########################################
    ############### Nodes Level ##############
    ##########################################    
    
    # Nodes Degree (althogh it is calculated in 'bipartite' package, it ignores zero degrees)
    if 'degree' in features_list:
        degX, degY = bipartite.degrees(B, (top_nodes))
        features_dict["node_level"]["degree"] = dict(degX) | dict(degY) #combine two dicts
    
    #
    if 'latapy_clustering' in features_list:
        features_dict["node_level"]["latapy_clustering"] = bipartite.latapy_clustering(B)
    
    # Redundancy
    if 'node_redundancy' in features_list:
        nodes_redundancy = [node for node in B if node if len(B[node]) >= 2] # Take all nodes that meet the function's criteria
        features_dict["node_level"]["node_redundancy"] = bipartite.node_redundancy(B, nodes_redundancy)
    
    # Centrality
    if 'betweenness_centrality' in features_list:
        features_dict["node_level"]["betweenness_centrality"] = bipartite.betweenness_centrality(B, nodes = top_nodes)
    
    if 'degree_centrality' in features_list:
        features_dict["node_level"]["degree_centrality"] = bipartite.degree_centrality(B, nodes = top_nodes)
    
    if 'closeness_centrality' in features_list:
        closeness_centrality = bipartite.closeness_centrality(B, nodes = top_nodes)
        features_dict["node_level"]["closeness_centrality"] = {k: v for k, v in closeness_centrality.items() if type(k) == str} # Remove numeric keys
    
    # Assortativity
    if 'average_neighbor_degree' in features_list:
        features_dict["node_level"]["average_neighbor_degree"] = nx.average_neighbor_degree(B)
    
    # PageRank
    if 'pagerank' in features_list:
        features_dict["node_level"]["pagerank"] = nx.pagerank(B)
    
    # HITS hubs and authorities
    # if any(s.startswith("hits") for s in features_list):
    #     h, a = nx.hits(B)
    # if 'hits_hubs' in features_list:
    #     features_dict["node_level"]["hits_hubs"] = h
    # if 'hits_authorities' in features_list:
    #     features_dict["node_level"]["hits_authorities"] = a    
    
    # Isolates
    if 'isolate' in features_list:
        features_dict["node_level"]["isolate"] = {isolate:1 for isolate in list(nx.isolates(B))}
    
    #########################################
    ############### Link Level ##############
    #########################################
    
    #! link level features should be calculated while removing edges upon calculations are made
    
    # # Bridges
    # if 'possible_bridge' in features_list:
    #     features_dict["link_level"]["possible_bridge"] = getBridges(B, bottom_nodes, non_edges)
    
    ## Link Prediction
            
    ### 1. Deal with class == 0 (non-exsiting links & missing link)
    
    # Preferential attachment
    # if 'preferential_attachment' in features_list:
    #     features_dict["link_level"]["preferential_attachment"] = list(nx.preferential_attachment(B, ebunch=non_edges))
    
    # Common neighbor centrality
    # if 'common_neighbor_centrality' in features_list:
    #     features_dict["link_level"]["common_neighbor_centrality"] = list(nx.common_neighbor_centrality(B, ebunch=non_edges)) #slow on big nets
    
    # Shortest Paths
    # if 'shortest_path_length' in features_list:
    #     features_dict["link_level"]["shortest_path_length"] = []
    # shortest_path_length = dict(nx.shortest_path_length(B))
    # features_dict["link_level"]["shortest_path_length"] = [(k, k2, v2) for k, v in shortest_path_length.items() for k2, v2 in v.items() if (k != k2) and ((k, k2) not in edges)] # O(in non_edges) > O(not in edges)
    
    # Number of shortest paths
    # if 'shortest_paths_count' in features_list:
    #     features_dict["link_level"]["shortest_paths_count"] = []
    
    # Friends measure
    # if 'friends_measure' in features_list:
    #     features_dict["link_level"]["friends_measure"] = []

    # Katz measure
    # if 'katz_measure_b0.005' in features_list:
    #     features_dict["link_level"]["katz_measure_b0.005"] = katz_measure(B, beta=0.005, bottom_nodes=bottom_nodes, top_nodes=top_nodes, non_edges=non_edges)

    # if 'katz_measure_b0.05' in features_list:
    #     features_dict["link_level"]["katz_measure_b0.05"] = katz_measure(B, beta=0.05, bottom_nodes=bottom_nodes, top_nodes=top_nodes, non_edges=non_edges)

    # Infomap communitites detection
    if 'same_community_infomap' or 'flow_infomap' in features_list:
        same_community, flow_dict = infomap_modules(B)
        if 'same_community_infomap' in features_list:
            features_dict["link_level"]["same_community_infomap"] = []
        if 'flow_infomap' in features_list:
            # features_dict["link_level"]["flow_infomap_bottom"] = []
            features_dict["link_level"]["flow_infomap_top"] = []

    ### 1.1 Deal with functions who cannot compute measures for given set of nodes
    for u, v in non_edges:

        # try:
            # if 'shortest_path_length' in features_list:
            #     shortest_path_length = nx.shortest_path_length(B, u, v)
            # if 'shortest_paths_count' in features_list:
            #     shortest_paths_count = len(list(nx.all_shortest_paths(B, u, v)))
        # except nx.NetworkXNoPath:
            # if 'shortest_path_length' in features_list:
            #     shortest_path_length = 0
            # if 'shortest_paths_count' in features_list:
            #     shortest_paths_count = 0                    
        
        # if 'shortest_path_length' in features_list:
        #     features_dict["link_level"]["shortest_path_length"] += [(u, v, shortest_path_length)]
        # if 'shortest_paths_count' in features_list:
        #     features_dict["link_level"]["shortest_paths_count"] += [(u, v, shortest_paths_count)]
        # if 'friends_measure' in features_list:
        #     features_dict["link_level"]["friends_measure"] += [(u, v, friends_measure(B, u, v))]
        if 'same_community_infomap' in features_list:
            features_dict["link_level"]["same_community_infomap"] += [next((tup for tup in same_community if tup[0] == u and tup[1] == v), None)]
            # features_dict["link_level"]["flow_infomap_bottom"] += [(u, v, flow_dict[u])]
            features_dict["link_level"]["flow_infomap_top"] += [(u, v, flow_dict[v])]
    
    ### 2. Deal with class == 1 (exsiting links)
    B_edge_remove = B.copy() # avoid remove_edge altering original object
    for u, v in edges:
        
        B_edge_remove.remove_edge(u, v)

        # if 'preferential_attachment' in features_list:
        #     features_dict["link_level"]["preferential_attachment"] += list(nx.preferential_attachment(B_edge_remove, ebunch=[(u,v)]))
        # if 'common_neighbor_centrality' in features_list:
        #     features_dict["link_level"]["common_neighbor_centrality"] += list(nx.common_neighbor_centrality(B_edge_remove, ebunch=[(u,v)])) #slow on big nets
        try:
            # if 'shortest_path_length' in features_list:
            #     shortest_path_length = nx.shortest_path_length(B_edge_remove, u, v)
            # if 'shortest_paths_count' in features_list:
            #     shortest_paths_count = len(list(nx.all_shortest_paths(B_edge_remove, u, v)))
        # except nx.NetworkXNoPath:
        #     if 'shortest_path_length' in features_list:
        #         shortest_path_length = 0
        #     if 'shortest_paths_count' in features_list:
        #         shortest_paths_count = 0
        
        # if 'shortest_path_length' in features_list:
        #     features_dict["link_level"]["shortest_path_length"] += [(u, v, shortest_path_length)]
        # if 'shortest_paths_count' in features_list:
        #     features_dict["link_level"]["shortest_paths_count"] += [(u, v, shortest_paths_count)]
        
        # if 'friends_measure' in features_list:
        #     features_dict["link_level"]["friends_measure"] += [(u, v, friends_measure(B_edge_remove, u, v))]
        # if 'katz_measure_b0.005' in features_list:
        #     features_dict["link_level"]["katz_measure_b0.005"] += [(u, v, katz_measure(B_edge_remove, beta=0.005, node_u=u, node_v=v))]
        # if 'katz_measure_b0.05' in features_list:
        #     features_dict["link_level"]["katz_measure_b0.05"] += [(u, v, katz_measure(B_edge_remove, beta=0.05, node_u=u, node_v=v))]
        
        # if 'same_community_infomap' in features_list:
            # same_community, flow_dict = infomap_modules(B)
            # features_dict["link_level"]["same_community_infomap"] += [next((tup for tup in same_community if tup[0] == u and tup[1] == v), None)]
            # features_dict["link_level"]["flow_infomap_bottom"] += [(u, v, flow_dict[u])]
            # features_dict["link_level"]["flow_infomap_top"] += [(u, v, flow_dict[v])]
        
        B_edge_remove.add_edge(u, v)
    
    
    # Motifs
    # if 'motifs' in features_list:
    #     shared_motifs = getMotifs(B, bottom_nodes, top_nodes, max_motif_size = 4)
    #     features_dict["link_level"].update(shared_motifs)

    # remove empty dicts 
    features_dict = {k: v for k, v in features_dict.items() if v}

    subsample = dict2df(edgelist_df, features_dict, top_nodes, bottom_nodes)
    
    return subsample

# A function to handle arguments for parallel processing
def helper_function(args, log_time=True):

    subsample, features_list = args

    if log_time:
        logging.basicConfig(filename='logs/features_extraction_py.log', level=logging.INFO)
        logging.info('Subsample: %s', str(subsample['subsample_ID'].iloc[0]))
        start_time = time.time()

    features = getNetFeatures(subsample, features_list)

    if log_time:
        logging.info('Size: %s', features['size'].iloc[0])
        logging.info('Time: %s seconds\n', "{:.2f}".format(time.time() - start_time))

    return features

def extract_features(df, features_list, nslots=1, table_path='data/processed/features_py.csv'):
    
    list_of_dfs = list(dict(tuple(df.groupby('subsample_ID'))).values())

    if nslots>1:
        
        pool = multiprocessing.Pool(processes=nslots)
        
        # Create a list of tuples (edgelist, features_list)
        args_list = [(edgelist, features_list) for edgelist in list_of_dfs]

        for edgelist_new in tqdm(pool.imap(helper_function, args_list), total=len(list_of_dfs), desc='Extracting network features (parallel processing)'):
            
            edgelist_new.to_csv(table_path, index=False, mode='a', header=not pd.io.common.file_exists(table_path))
            
        pool.close()
        pool.join()

    else:
        
        for edgelist in tqdm(list_of_dfs, desc='Extracting network features (single process)'):
            
            edgelist_new = getNetFeatures(edgelist, features_list)
            edgelist_new.to_csv(table_path, index=False, mode='a', header=not pd.io.common.file_exists(table_path))
    
    print('Done', flush=True)