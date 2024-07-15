# Import Packages

import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import itertools

import networkx as nx
from networkx.algorithms import bipartite

import multiprocessing
from tqdm import tqdm # Progress Bar

from helper.base import pandas2bigraph
# from helper.motifs_functions import getMotifs

####################################################################################

logging_enabled = False

# Decorator for logging time
def log_time(should_log=False):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            if logging_enabled and should_log:
                # Log the duration immediately
                logging.basicConfig(filename='logs/features_extraction_features_py.log', level=logging.INFO, filemode='a')
                subsample_id = kwargs.get('subsample_ID', 'Unknown')
                logging.info('Subsample: %s, Feature: %s, Duration: %.4f seconds', subsample_id, func.__name__, duration)

            return result
        return wrapper
    return decorator

# For logging in general
# logging.basicConfig(filename='logs/errors.log', level=logging.INFO, filemode='a')
# logging.info(something)

####################################################################################


############################################
############### Network Level ##############
############################################

@log_time(should_log=False)
def calculate_network_size(B, top_nodes, bottom_nodes, **kwargs):
    return len(top_nodes) + len(bottom_nodes)

@log_time(should_log=False)
def calculate_species_ratio(B, top_nodes, bottom_nodes, **kwargs):
    return len(top_nodes) / len(bottom_nodes)

@log_time(should_log=False)
def calculate_interactions_count(B, top_nodes, bottom_nodes, **kwargs):
    return B.size(weight=None)

@log_time(should_log=False)
def calculate_edge_connectivity(B, top_nodes, bottom_nodes, **kwargs):
    return nx.edge_connectivity(B)

@log_time(should_log=False)
def calculate_density(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.density(B, top_nodes)

@log_time(should_log=False)
def calculate_bipartite_clustering(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.robins_alexander_clustering(B)

@log_time(should_log=False)
def calculate_Spectral_bipartivity(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.spectral_bipartivity(B)

@log_time(should_log=False)
def calculate_average_clustering(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.average_clustering(B)

@log_time(should_log=False)
def calculate_degree_assortativity_coefficient(B, top_nodes, bottom_nodes, **kwargs):
    return nx.degree_assortativity_coefficient(B)

@log_time(should_log=False)
def calculate_global_efficiency(B, top_nodes, bottom_nodes, **kwargs):
    return nx.global_efficiency(B)

@log_time(should_log=False)
def calculate_local_efficiency(B, top_nodes, bottom_nodes, **kwargs):
    return nx.local_efficiency(B)

@log_time(should_log=False)
def calculate_connected_components(B, top_nodes, bottom_nodes, **kwargs):
    return nx.number_connected_components(B)

##########################################
############### Nodes Level ##############
##########################################  

@log_time(should_log=False)
def calculate_degree(B, top_nodes, bottom_nodes, **kwargs):
    degX, degY = bipartite.degrees(B, (top_nodes))
    return dict(degX) | dict(degY) #combine two dicts

@log_time(should_log=False)
def calculate_latapy_clustering(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.latapy_clustering(B)

@log_time(should_log=False)
def calculate_node_redundancy(B, top_nodes, bottom_nodes, **kwargs):
    nodes_redundancy = [node for node in B if node if len(B[node]) >= 2] # Take all nodes that meet the function's criteria
    return bipartite.node_redundancy(B, nodes_redundancy)

@log_time(should_log=False)
def calculate_betweenness_centrality(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.betweenness_centrality(B, nodes = top_nodes)

@log_time(should_log=False)
def calculate_degree_centrality(B, top_nodes, bottom_nodes, **kwargs):
    return bipartite.degree_centrality(B, nodes = top_nodes)

@log_time(should_log=False)
def calculate_closeness_centrality(B, top_nodes, bottom_nodes, **kwargs):
    closeness_centrality = bipartite.closeness_centrality(B, nodes = top_nodes)
    return {k: v for k, v in closeness_centrality.items() if type(k) == str} # Remove numeric keys

@log_time(should_log=False)
def calculate_average_neighbor_degree(B, top_nodes, bottom_nodes, **kwargs):
    return nx.average_neighbor_degree(B)

@log_time(should_log=False)
def calculate_pagerank(B, top_nodes, bottom_nodes, **kwargs):
    return nx.pagerank(B)

@log_time(should_log=False)
def calculate_hits_hubs(B, top_nodes, bottom_nodes, **kwargs):
    h, a = nx.hits(B)
    return h

@log_time(should_log=False)
def calculate_hits_authorities(B, top_nodes, bottom_nodes, **kwargs):
    h, a = nx.hits(B)
    return a

@log_time(should_log=False)
def calculate_isolate(B, top_nodes, bottom_nodes, **kwargs):
    return {isolate:1 for isolate in list(nx.isolates(B))}

#########################################
############### Link Level ##############
#########################################

 #! link level features should be calculated while removing edges upon calculations are made
def edge_features_helper(B, top_nodes, bottom_nodes, func, edges, non_edges=None):
    result = []
    
    # 1.1 Deal with functions who cannot compute measures for given set of nodes
    if non_edges != None:
        for u, v in non_edges:
            try:
                value = func(B, u, v)
            except nx.NetworkXNoPath:
                value = [(u, v, 0)]
            
            result += value

    # 2. Deal with class == 1 (existing links)
    B_edge_remove = B.copy() # avoid remove_edge altering original object

    for u, v in edges:
        B_edge_remove.remove_edge(u, v)

        try:
            value = func(B_edge_remove, u, v)
        except nx.NetworkXNoPath:
            value = [(u, v, 0)]

        result += value

        B_edge_remove.add_edge(u, v)
    
    return result

def bridges_helper(B, bottom_nodes, non_edges):
    
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

def preferential_attachment_helper(B, u, v):
    return list(nx.preferential_attachment(B, ebunch=[(u,v)]))



def shortest_path_length_helper(B, u, v):
    result = nx.shortest_path_length(B, u, v)
    return [(u, v, result)]

def shortest_paths_count_helper(B, u, v):
    result = len(list(nx.all_shortest_paths(B, u, v, method='bfs'))) # bfs seems to be faster than Dijkstra
    return [(u, v, result)]

# Friends measure
def friends_measure_helper(G, u, v):
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

    score = 0
    for x in G[u]:
        for y in G[v]:
            if G.has_edge(x,y): # or x == y | dropped as it will be used only for bipartite
                score = score + 1
    return [(u, v, score)]

# Katz measure
def katz_measure_helper(G, node_u=None, node_v=None, beta=0.05, non_edges=[], bottom_nodes=None, top_nodes=None):
    # Get the adjacency matrix of the graph
    A = nx.adjacency_matrix(G)
    
    # Calculate the identity matrix
    I = np.eye(G.number_of_nodes())
    
    # Compute the Katz Index
    # Katz(u, v) = (I - beta * A)^-1 - I
    try: 
        katz_matrix = np.linalg.pinv(I - beta * A) - I
        # Katz_matrix = np.linalg.inv(I - beta * A) - I
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
        return [(node_u, node_v, katz_value)]
    
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

def bigraph_relable_to_integer(B, return_sep=False, return_mapping=False):

    bottom_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]
    top_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 1]

    mapping = {name : i+1 for i, name in enumerate(top_nodes+bottom_nodes)}

    B = nx.relabel_nodes(B, mapping)

    if return_sep or return_mapping:
        return_tuple = [B]
        if return_sep:
            return_tuple += [len(top_nodes)+1]
        if return_mapping:
            return_tuple += [mapping]

        return tuple(return_tuple)

    return B

def infomap_helper(B, u=None, v=None, return_df=False):

    from infomap import Infomap
    
    B_numerical_id, bipartite_start_id, nodes_mapping = bigraph_relable_to_integer(B, return_sep=True, return_mapping=True) # Relable nodes to integers
    im = Infomap(two_level=True, flow_model='undirected', skip_adjust_bipartite_flow=True, silent=True) # Create Infomap object
    im.add_networkx_graph(B_numerical_id) # Add networkx graph to Infomap object
    im.bipartite_start_id = bipartite_start_id # Set the start id for bipartite nodes
    im.run() # Run the Infomap search algorithm to find optimal modules
    
    # start_time = time.time()
    if u and v:
        
        u_id = nodes_mapping[u]
        v_id = nodes_mapping[v]
        
        u_node = next((n for n in im.nodes if n.node_id == u_id), None)
        v_node = next((n for n in im.nodes if n.node_id == v_id), None)

        same_community = [(u, v, int(u_node.module_id == v_node.module_id))]
        flow_dict = {v:v_node.flow, u:u_node.flow}
        modular_centrality_dict = {v:v_node.modular_centrality+0, u:u_node.modular_centrality+0} # "+0" eliminate -0.0
        return [(same_community, flow_dict, modular_centrality_dict)]

    
    modules_df = im.get_dataframe(columns=["node_id", "module_id", "flow", "modular_centrality"]) # Get the modules as a pandas dataframe
    modules_df['node_name'] = [B_numerical_id.nodes[node_id]['name'] for node_id in modules_df['node_id']]  # Add the node names as a column
    modules_df['bipartite'] = [B_numerical_id.nodes[node_id]['bipartite'] for node_id in modules_df['node_id']]

    modules_df['node_name'] = modules_df['node_name']

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
    modular_centrality_dict = dict(zip(modules_df['node_name'], modules_df['modular_centrality']))
    
    return same_community, flow_dict, modular_centrality_dict

@log_time(should_log=False)
def calculate_preferential_attachment(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')
    result = list(nx.preferential_attachment(B, ebunch=non_edges))
    result += edge_features_helper(B, top_nodes, bottom_nodes, preferential_attachment_helper, edges, non_edges=None)
    return result

@log_time(should_log=False)
def calculate_shortest_path_length(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')
    return edge_features_helper(B, top_nodes, bottom_nodes, shortest_path_length_helper, edges, non_edges)

@log_time(should_log=False)
def calculate_shortest_paths_count(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')
    return edge_features_helper(B, top_nodes, bottom_nodes, shortest_paths_count_helper, edges, non_edges)

@log_time(should_log=False)
def calculate_friends_measure(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')
    return edge_features_helper(B, top_nodes, bottom_nodes, friends_measure_helper, edges, non_edges) 

@log_time(should_log=False)
def calculate_katz_measure(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')
    result = katz_measure_helper(B, beta=0.05, bottom_nodes=bottom_nodes, top_nodes=top_nodes, non_edges=non_edges)
    result += edge_features_helper(B, top_nodes, bottom_nodes, katz_measure_helper, edges, non_edges=None) # , beta=0.05
    return result

@log_time(should_log=False)
def calculate_same_community_infomap(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')

    same_community, flow_dict, modular_centrality_dict = infomap_helper(B)

    same_community_infomap = []
    # flow_infomap = {}

    for u, v in non_edges: # not used in edge_features_helper as it is not efficient
        same_community_infomap += [next((tup for tup in same_community if tup[0] == u and tup[1] == v), None)]
        flow_dict.update(flow_dict)
        modular_centrality_dict.update(modular_centrality_dict)
    
    result = edge_features_helper(B, top_nodes, bottom_nodes, infomap_helper, edges, non_edges=None)

    for i in result:
        same_community_infomap += i[0]
        flow_dict.update(i[1])
        modular_centrality_dict.update(i[2])

    return same_community_infomap, flow_dict, modular_centrality_dict

@log_time(should_log=False)
def calculate_possible_bridge(B, top_nodes, bottom_nodes, **kwargs):
    edges, non_edges = kwargs.get('edges'), kwargs.get('non_edges')
    return bridges_helper(B, bottom_nodes, non_edges)


####################################################################################
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
    if features_dict['network_level']:
        network_level = features_dict["network_level"]
        network = network.assign(**network_level)
    
    # Add nodes-level features to edge list
    if features_dict['node_level']:
        node_level = pd.concat([pd.DataFrame({key: val}) for key, val in features_dict["node_level"].items()], axis=1)
        network = network.merge(node_level.add_suffix('_LL'), left_on='lower_level', right_index=True, how = 'left')
        network = network.merge(node_level.add_suffix('_HL'), left_on='higher_level', right_index=True, how = 'left')
    
    # Add link-level features to edge list
    if features_dict['link_level']:
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

    
    # Drop columns which are not new features
    network = network.drop(drop_columns, axis = 1)

    # Rounding all float columns to 4 decimal places
    network = network.round(4)
    
    return network

def getNetFeatures(edgelist_df, features_list=[], log_time=True):
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
    
    # Convert the edgelist to a networkx graph
    B = pandas2bigraph(edgelist_df)
    
    # Get the subsample id
    subsample_id = edgelist_df['subsample_ID'].iloc[0]

    ## get top and bottom nodes
    bottom_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]
    top_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 1]
    
    # Get existing, non-existing and all possible edges
    edgelist_minimal = edgelist_df[edgelist_df['weight'] != 0][['lower_level', 'higher_level']] # TODO: do I need this?
    possible_edges = list(itertools.product(bottom_nodes, top_nodes))
    edges = list(zip(edgelist_minimal.lower_level, edgelist_minimal.higher_level))
    non_edges = list(set(possible_edges).difference(set(edges)))
    
    # Define a dictionary of all features and their corresponding functions
    feature_functions = {
        'network_size': (calculate_network_size, 'network_level'),
        'species_ratio': (calculate_species_ratio, 'network_level'),
        'interactions_count': (calculate_interactions_count, 'network_level'),
        'edge_connectivity': (calculate_edge_connectivity, 'network_level'),
        'density': (calculate_density, 'network_level'),
        'bipartite_clustering': (calculate_bipartite_clustering, 'network_level'),
        'Spectral_bipartivity': (calculate_Spectral_bipartivity, 'network_level'),
        'average_clustering': (calculate_average_clustering, 'network_level'),
        'degree_assortativity_coefficient': (calculate_degree_assortativity_coefficient, 'network_level'),
        'global_efficiency': (calculate_global_efficiency, 'network_level'),
        'local_efficiency': (calculate_local_efficiency, 'network_level'),
        'connected_components': (calculate_connected_components, 'network_level'),
        'degree': (calculate_degree, 'node_level'),
        'latapy_clustering': (calculate_latapy_clustering, 'node_level'),
        'node_redundancy': (calculate_node_redundancy, 'node_level'),
        'betweenness_centrality': (calculate_betweenness_centrality, 'node_level'),
        'degree_centrality': (calculate_degree_centrality, 'node_level'),
        'closeness_centrality': (calculate_closeness_centrality, 'node_level'),
        'average_neighbor_degree': (calculate_average_neighbor_degree, 'node_level'),
        'pagerank': (calculate_pagerank, 'node_level'),
        'hits_hubs': (calculate_hits_hubs, 'node_level'),
        'hits_authorities': (calculate_hits_authorities, 'node_level'),
        'isolate': (calculate_isolate, 'node_level'),
        'preferential_attachment': (calculate_preferential_attachment, 'link_level'),
        'shortest_path_length': (calculate_shortest_path_length, 'link_level'),
        'shortest_paths_count': (calculate_shortest_paths_count, 'link_level'),
        'friends_measure': (calculate_friends_measure, 'link_level'),
        'katz_measure': (calculate_katz_measure, 'link_level'),
        'same_community_infomap': (calculate_same_community_infomap, 'link_level'),
        'flow_infomap': (None, 'node_level'),
        'modular_centrality_infomap': (None, 'node_level'),
        'possible_bridge': (calculate_possible_bridge, 'link_level'),
        # 'motifs': (calculate_motifs, 'link_level'),
    }

    # functions of features that return more than one value
    features_multi_values = { 
        'same_community_infomap': ('same_community_infomap', 'flow_infomap', 'modular_centrality_infomap')
    }

    # Create an empty dictionary to store the features
    features_dict = {}

    # Create empty dictionaries for each scope
    features_dict = {'network_level': {}, 'node_level': {}, 'link_level': {}}

    for feature in features_list:

        kwargs = {}

        if log_time:
            kwargs.update({'subsample_ID': subsample_id})

        if feature in feature_functions:
            try:
                
                # Get the function and the scope of the feature
                func = feature_functions[feature][0]
                level = feature_functions[feature][1]

                # Create empty dictionaries for each scope
                if level not in features_dict:
                    features_dict[level] = {}
                    if feature in features_multi_values: # Check if the feature returns more than one value
                        for feature_ in features_multi_values[feature]:
                            level_ = features_multi_values[feature_][1]
                            if level_ not in features_dict:
                                features_dict[level_] = {}

                # Add the edges and non-edges to the kwargs if the feature is a link-level feature
                if level == 'link_level':
                    kwargs.update({'edges': edges, 'non_edges': non_edges})

                # Call the appropriate function and store the result
                feature_result = func(B, top_nodes, bottom_nodes, **kwargs)

                if feature_result is None:
                    raise Exception(f"Feature {feature} returned None")
                elif type(feature_result) == tuple:
                    for i, feature_ in enumerate(features_multi_values[feature]):
                        level_ = feature_functions[feature_][1]
                        features_dict[level_][feature_] = feature_result[i]
                else:
                    features_dict[level][feature] = feature_result

            except Exception as e:
                print(f"Error calculating feature {feature}: {e}")
        else:
            raise Exception(f"Feature {feature} not found in feature_functions")

    # Convert the dictionary to a dataframe
    subsample = dict2df(edgelist_df, features_dict, top_nodes, bottom_nodes)
    
    return subsample

# A function to handle arguments for parallel processing and logging
def helper_function(args, log_time=False):

    subsample, features_list = args

    if log_time:
        logging.basicConfig(filename='logs/features_extraction_subsamples_py.log', level=logging.INFO)
        logging.info('Subsample: %s', str(subsample['subsample_ID'].iloc[0]))
        start_time = time.time()

    features = getNetFeatures(subsample, features_list)

    if log_time:
        logging.info('Size: %s', features['network_size'].iloc[0])
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