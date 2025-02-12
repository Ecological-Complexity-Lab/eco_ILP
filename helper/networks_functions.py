import os
import numpy as np
import pandas as pd
import glob # For database lookup
from tqdm import tqdm #Progress Bar
import networkx as nx
from itertools import product

from helper.base import pandas2bigraph

####################################################################################

    
def load_networks(path, ignored_communities=[]):

    # TODO: Implement an efficient way to load the networks, without dictionaries of matrices and edge lists..
    
    domain_mapping = {
        "Journal":"Journal",
        "Oikos":"Journal",
        "American_Naturalist":"Journal",
        "Ecography":"Journal",
        "Ecological_Applications":"Journal",
        "Ecological_Monographs":"Journal",
        "Ecology":"Journal",
        "Chicago":"Chicago",
        "Denver":"Denver",
        "Minneapolis":"Minneapolis",
        "San_Francisco":"San Francisco",
        "Washington":"Washington",
        "US_House":"Legislature",
        "US_Senate":"Legislature",
        "UN":"Legislature",
        "EP":"Legislature",
        "HMP":"Microbiome",
        "Act":"Actor",
        "baseball":"Baseball",
        "basketball":"Basketball",
        "hockey":"Hockey",
        "M_AP":"Plant-Ant",
        "A_HP":"Host-Parasite",
        "A_PH":"Plant-Herbivore",
        "M_PL":"Plant-Pollinator",
        "M_SD":"Plant-Seed Dispersers"
    }


    # Initiate a new dictionary that will contain the new networks
    networks = {}
    
    # Get path to dataframes
    dataframes_path = glob.glob(path)

    # Set files to keep
    keep_files = []

    # First go through the files
    for file in tqdm(dataframes_path, desc='Loading networks'):
        
        path_split = file.split(os.sep) # Split the path name
        file_name = path_split[-1] # Get file name

        if ('dictionary' in file_name) or ('graphletcounts' in file_name) or ('signatures' in file_name):
            continue

        network_name = file_name.split(".")[0] # Get network name
        network_domain = next((domain_mapping[key] for key in domain_mapping if network_name.startswith(key)), None) # Get the domain of the network

        if network_domain in ignored_communities:
            continue

        keep_files.append(file)
    
    # Load networks
    for file in tqdm(keep_files, desc='Loading networks'):
        
        path_split = file.split(os.sep) # Split the path name
        file_name = path_split[-1] # Get file name
        network_name = file_name.split(".")[0] # Get network name
        network_domain = next((domain_mapping[key] for key in domain_mapping if network_name.startswith(key)), None) # Get the domain of the network

        net_edgelist = pd.read_csv(file, sep='\s+', engine='python', names=['lower_level','higher_level', 'weight'], header=None, index_col=False, encoding = "ISO-8859-1") # Reading the file | "sep='\s+', engine='python'" is used to handle multiple spaces as separators
        net_edgelist['weight'] = 1 # Adding weight column
        net_edgelist['lower_level'] = net_edgelist['lower_level'].astype(str)
        net_edgelist['higher_level'] = net_edgelist['higher_level'].astype(str)
        # net_graphletcounts = pd.read_csv(file+'.graphletcounts.txt', sep=' ', names=['graphlet_id','counts'], header=None, index_col=False) # Reading the file
        # net_dictionary = pd.read_csv(file+'.dictionary.txt', sep=' ', names=['node_id','node_name'], header=None, index_col=False) # Reading the file
        # net_signatures = pd.read_csv(file+'.signatures.txt', sep=' ', header=None, index_col=False) # Reading the file

        B = pandas2bigraph(net_edgelist)
        
        # Get the node names
        bottom_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 0]
        top_nodes = [n for n in B.nodes if B.nodes[n]['bipartite'] == 1]

        # Convert the adjacency matrix to a DataFrame
        net_matrix = pd.DataFrame(nx.bipartite.biadjacency_matrix(B, bottom_nodes, top_nodes).todense(), index=bottom_nodes, columns=top_nodes)
        
        # Add non-edges to the edgelist
        unique_top_nodes = net_edgelist['higher_level'].unique()  # Getting unique nodes
        unique_bottom_nodes = net_edgelist['lower_level'].unique()
        all_combinations = pd.DataFrame(product(unique_top_nodes, unique_bottom_nodes), columns=['higher_level', 'lower_level']) # Creating all possible combinations
        net_edgelist = all_combinations.merge(net_edgelist, on=['higher_level', 'lower_level'], how='left').fillna({'weight': 0}) # Merging with the existing edgelist and filling missing weights with 0

        # Combine relevant information about the layer into a dict
        layers_data = {"Adjacency_Matrix":net_matrix,
                        "Edge_List":net_edgelist,
                        "Subsamples":{}}
        
        # Insert the network including meta data to 'networks' dictionary
        networks[network_name] = {"Network_Layers":{1:layers_data},
                                    "community":network_domain,
                                    "Type":"Monolayer",
                                    "Ref":""}
        
    return networks

####################################################################################

def sample(adj_matrix, frac, weighted, degree_biased=None):
    '''
    generate a single sample.

    Parameters
    ----------
    adj_matrix : pd.DataFrame
        Adjacency matrix.
    frac : float
        Fraction of the desired observed network.
    weighted : bool, optional
        True if using edge weights. The default is False.
    degree_biased : str, optional
        'high' or 'low' if the sampling should be biased towards high or low degree nodes. The default is None.

    Returns
    -------
    network_sample : pd.DataFrame
        Adjacency matrix as 'network' argument, with removed links according to 'frac' argument.

    '''
    rows, cols = adj_matrix.shape
    adj_matrix_np = adj_matrix.to_numpy()
    total_links = int(adj_matrix_np.sum() * frac)
    
    if weighted:
        # Weighted sampling logic remains the same
        weights_sum = round(adj_matrix_np.sum() * frac, 0)
        network_flat = adj_matrix_np.flatten()
        network_probs = network_flat / network_flat.sum()
        network_sample = np.random.multinomial(n=int(weights_sum), pvals=network_probs, size=1)
        
    else:
        # Get indices of existing links
        existing_links = np.where(adj_matrix_np > 0)
        link_indices = list(zip(existing_links[0], existing_links[1]))
        
        if degree_biased:

            # Calculate degrees
            row_degrees = adj_matrix_np.sum(axis=1)
            col_degrees = adj_matrix_np.sum(axis=0)
            
            # Calculate degree-based probabilities
            if degree_biased == 'high':
                # Probability proportional to degree
                link_probs = np.array([
                    (row_degrees[i] + col_degrees[j]) 
                    for i, j in link_indices
                ])
            elif degree_biased == 'low':
                # Probability inversely proportional to degree
                link_probs = np.array([
                    1 / (row_degrees[i] + col_degrees[j] + 1)  # Add 1 to avoid division by zero
                    for i, j in link_indices
                ])
            
            # Normalize probabilities
            link_probs = link_probs / link_probs.sum()
            
        else:
            # Uniform sampling among existing links
            link_probs = np.ones(len(link_indices)) / len(link_indices)
        
        # Sample links based on calculated probabilities
        sampled_indices = np.random.choice(
            len(link_indices), 
            size=total_links, 
            replace=False, 
            p=link_probs
        )
        
        # Create the sampled network
        network_sample = np.zeros((rows, cols), dtype=int)
        for idx in sampled_indices:
            i, j = link_indices[idx]
            network_sample[i, j] = 1
    
    return pd.DataFrame(network_sample, index=adj_matrix.index, columns=adj_matrix.columns)


    
####################################################################################

def create_samples(adj_matrix, edgelist, frac_list, reps = 1, weighted = False, min_components = False, degree_biased = None):
    '''
    Generating subsamples.

    Parameters
    ----------
    adj_matrix : pd.DataFrame
        Adjency matrix.
    edgelist : pd.DataFrame
        Edge list.
    frac_list : list
        list of desired fractions.
    reps : int, optional
        number of repetition of subsampling. The default is 1.
    weighted : bool, optional
        True if using egde weights. The default is False.
    min_components : bool, optional
        True if the algorithm should minimize the number of components of each generated sample.
    
    Returns
    -------
    subsamples : dict
        Containing all generated subsamples

    '''

    # TODO: Implement a more efficient code, so only an edge list will be imported
    # TODO: Change terminology, so instead of 'TP', 'TN' and 'FN' (True-Positive link, True-Negative link..), it will be 1, 0, and -1 (representing 'Observed', 'Non-observed' and 'Missed')
    
    if weighted == False: 
        adj_matrix = adj_matrix > 0 # convert weighted networks to binary
        adj_matrix = adj_matrix.astype(int)
    
    # Create a dictionary storing all subsamples - for each fraction
    subsamples = {frac:{} for frac in frac_list+[1]}
    
    B = pandas2bigraph(edgelist) # Convert pandas dataframe to networkx bipartite graph
        
    for frac in frac_list:
                
        for rep in range(1, reps+1):
            
            if min_components:
                
                subsamples_dict = {} # Create empty dict containing subsampled networks
                
                for i in range(2): # Generate subsamples, redo the random sampling if subsample is not connected
                    adj_matrix_sample = sample(adj_matrix, frac, weighted, degree_biased)
                    edgeList_sample = adj_matrix_sample.stack().rename_axis(['lower_level','higher_level']).rename('weight').reset_index() #Create an edge list
                    B_sample = pandas2bigraph(edgeList_sample)
                    
                    subsamples_dict[nx.number_connected_components(B_sample)] = (adj_matrix_sample, edgeList_sample)
                    if nx.number_connected_components(B_sample) == nx.number_connected_components(B):
                        break
                    
                adj_matrix_sample, edgeList_sample = subsamples_dict[min(subsamples_dict.keys())]
                
            else:
                adj_matrix_sample = sample(adj_matrix, frac, weighted, degree_biased)
                edgeList_sample = adj_matrix_sample.stack().rename_axis(['lower_level','higher_level']).rename('weight').reset_index() #Create an edge list
            
            #Set a class for each link, wether its TP, TN or Missed Obs
            TP = (adj_matrix > 0) & (adj_matrix_sample > 0) # TP link = existing link = 1
            TN = (adj_matrix == 0) & (adj_matrix_sample == 0) # TN link = non-existing link = 0
            FN = (adj_matrix > 0) & (adj_matrix_sample == 0) # Missed Obs = subsampled link = -1
            
            edgeList_sample = pd.merge(edgeList_sample, TP.stack().rename_axis(['lower_level','higher_level']).rename('TP').reset_index(), how='left', on=['lower_level', 'higher_level'])
            edgeList_sample = pd.merge(edgeList_sample, TN.stack().rename_axis(['lower_level','higher_level']).rename('TN').reset_index(), how='left', on=['lower_level', 'higher_level'])
            edgeList_sample = pd.merge(edgeList_sample, FN.stack().rename_axis(['lower_level','higher_level']).rename('FN').reset_index(), how='left', on=['lower_level', 'higher_level'])
        
            edgeList_sample.loc[edgeList_sample.TP == True, 'class'] = 1
            edgeList_sample.loc[edgeList_sample.TN == True, 'class'] = 0
            edgeList_sample.loc[edgeList_sample.FN == True, 'class'] = -1
            
            edgeList_sample = edgeList_sample.drop(["TP", "TN", "FN"], axis=1)
            
            subsamples[frac][rep] = {"Adjacency_Matrix":adj_matrix_sample,
                                     "Edge_List":edgeList_sample}
    
    
    # Add also frac = 1 (Original network)
    edgelist['class'] = edgelist.apply(lambda row: 1 if row['weight'] > 0 else 0, axis=1)
    
    subsamples[1][1] = {"Adjacency_Matrix":adj_matrix,
                          "Edge_List":edgelist}
    
    return subsamples

####################################################################################

def process_networks(networks, minNetworkSize = 20, maxNetworkSize = 1000, minConnectance = 0.1, frac_list = [0.8], reps=1, min_components=False, weighted=False, degree_biased=None, reverse_filters=False):
    processed_networks = {}
    # filtered_networks = {}

    if degree_biased != None and weighted == True:
        raise ValueError("Degree biased sampling is currently only available for binary networks.")
    
    for net_id, net_data in tqdm(networks.items(), desc='Processing and subsampling'):
        processed_layers = {}
        community = net_data['community']
        
        for layer, layer_data in net_data["Network_Layers"].items():
            
            adj_matrix = layer_data['Adjacency_Matrix']
            edge_list = layer_data['Edge_List']
            
            connectanceFilter = False
            sizeFilter = False
            networkSize = sum(adj_matrix.shape)
            connectance = np.count_nonzero(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1])
            
            # Check network size
            if not (minNetworkSize <= networkSize <= maxNetworkSize):
                sizeFilter = True
            
            # Check connectance
            if connectance < minConnectance:
                connectanceFilter = True
                
            if (sizeFilter or connectanceFilter) and not reverse_filters:
                # filtered_networks[net_id] = {'community': community, 'net_size': networkSize, 'connectance': connectance, 'connectance_filtered':connectanceFilter, 'net_size_filtered': sizeFilter}
                continue
            elif not (sizeFilter or connectanceFilter) and reverse_filters:
                continue
            
            # Generate subsamples
            sample = create_samples(adj_matrix, edge_list, frac_list, reps=reps, weighted=weighted, min_components=min_components, degree_biased=degree_biased)
            
            # Create new layer data
            new_layer_data = layer_data.copy()
            new_layer_data["Subsamples"] = sample
            
            processed_layers[layer] = new_layer_data
        
        # Only add network if it has layers
        if processed_layers:
            processed_networks[net_id] = {
                **net_data,
                "Network_Layers": processed_layers
            }
    
    return processed_networks#, filtered_networks

####################################################################################

def export2csv(networks, output_dir, file_prefix='subsamples'):
    
    '''
    Exporting the proccessed subsamples to csv file.

    Parameters
    ----------
    networks : dict
        Contains all networks, including subsamples.
    output_dir : str
        directory for the new csv file(s).

    Returns
    -------
    None.

    '''
        
    subsample_id = 0
    link_id_counter = 0
    
    for net_id, net_data in tqdm(networks.items(), desc='Exporting to csv'):
        
        for layer, layer_data in net_data["Network_Layers"].items():
            
            for frac, subsamples in layer_data["Subsamples"].items():
                
                for rep, subsample in subsamples.items():
                    
                    network = subsample["Edge_List"].copy()
                    
                    # Insert generated id to each subsample
                    subsample_id += 1
                    network.insert(0, 'subsample_ID', subsample_id)
                    
                    # Assign and increment link_ID
                    network = network.reset_index(drop=False).rename(columns={'index': 'link_ID'})
                    network['link_ID'] += link_id_counter
                    link_id_counter += len(network)  # Update the link_id_counter

                    # Append subsample to subsamples_edge_lists.csv
                    network.to_csv(output_dir+file_prefix+'_edge_lists.csv', index=False, mode='a', header=not pd.io.common.file_exists(output_dir+file_prefix+'_edge_lists.csv'))
                    
                    # Save network descriptions to subsamples_meta dataframe 
                    meta_data = pd.DataFrame({
                        'subsample_ID':subsample_id,
                        'name':net_id, 
                        'community':net_data['community'], 
                        # 'symbiotic_relationship':net_data['Symbiotic_Relationship'], 
                        'type':net_data['Type'], 
                        'layer':layer, 
                        'fraction':frac, 
                        'repetition':rep}, index=[0])
                    
                    meta_data.to_csv(output_dir+file_prefix+'_metadata.csv', index=False, mode='a', header=not pd.io.common.file_exists(output_dir+file_prefix+'_metadata.csv'))