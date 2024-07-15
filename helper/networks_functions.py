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
    
    # Load networks
    for file in tqdm(dataframes_path, desc='Loading networks', total=len(dataframes_path)/4):
        
        path_split = file.split(os.sep) # Split the path name
        file_name = path_split[-1] # Get file name

        if ('dictionary' in file_name) or ('graphletcounts' in file_name) or ('signatures' in file_name):
            continue

        network_name = file_name.split(".")[0] # Get network name
        network_domain = next((domain_mapping[key] for key in domain_mapping if network_name.startswith(key)), None) # Get the domain of the network

        if network_domain in ignored_communities:
            continue

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

def sample(network, frac, weighted):
    '''
    generate a single sample.

    Parameters
    ----------
    network : pd.DataFrame
        Adjency matrix.
    frac : float
        Fraction of the links that will be removed.
    weighted : bool, optional
        True if using egde weights. The default is False.

    Returns
    -------
    network_sample : pd.DataFrame
        Adjency matrix as 'network' argument, with removed links according to 'frac' argument.

    '''
    network_flat = network.to_numpy().flatten() # random.multinomial accept only 1d arrays
    
    if weighted == False:

        # Pick random obs interactions
        obs_idx = np.random.choice(a=network_flat.nonzero()[0], size=int(sum(network_flat)*frac), replace=False) 

        # Initiate empty matrix and fill observations with 'obs_idx'
        network_sample = np.zeros(network_flat.shape[0])
        network_sample[obs_idx]=1
        
    else:
        
        weights_sum = round(network.values.sum()*frac,0) # Desired number of weights (total)
        network_probs = network_flat/network_flat.sum() # calc [proba]
        network_sample = np.random.multinomial(n=weights_sum, pvals=network_probs, size=1) # Generate sample out of multinomial distribution
    
    network_sample = network_sample.reshape(network.shape[0], network.shape[1]) # Reshape back to matrix
    network_sample = pd.DataFrame(network_sample, index=network.index, columns=network.columns) # Change back to dataframe
    
    return network_sample

    
####################################################################################

def create_samples(network, network_edgelist, frac_list, reps = 1, weighted = False, min_components = False):
    '''
    Generating subsamples.

    Parameters
    ----------
    network : pd.DataFrame
        Adjency matrix.
    network_edgelist : pd.DataFrame
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
        network = network > 0 # convert weighted networks to binary
    
    # Create a dictionary storing all subsamples - for each fraction
    subsamples = {frac:{} for frac in frac_list+[1]}
    
    B = pandas2bigraph(network_edgelist) # Convert pandas dataframe to networkx bipartite graph
        
    for frac in frac_list:
                
        for rep in range(1, reps+1):
            
            if min_components:
                
                subsamples_dict = {} # Create empty dict containing subsampled networks
                
                for i in range(2): # Generate subsamples, redo the random sampling if subsample is not connected
                    network_sample = sample(network, frac, weighted)
                    network_sample_edgeList = network_sample.stack().rename_axis(['lower_level','higher_level']).rename('weight').reset_index() #Create an edge list
                    B_sample = pandas2bigraph(network_sample_edgeList)
                    
                    subsamples_dict[nx.number_connected_components(B_sample)] = (network_sample, network_sample_edgeList)
                    if nx.number_connected_components(B_sample) == nx.number_connected_components(B):
                        break
                    
                network_sample, network_sample_edgeList = subsamples_dict[min(subsamples_dict.keys())]
                
            else:
                network_sample = sample(network, frac, weighted)
                network_sample_edgeList = network_sample.stack().rename_axis(['lower_level','higher_level']).rename('weight').reset_index() #Create an edge list
            
            #Set a class for each link, wether its TP, TN or Missed Obs
            TP = (network > 0) & (network_sample > 0) #TP
            TN = (network == 0) & (network_sample == 0) #TN
            FN = (network > 0) & (network_sample == 0) #Missed Obs
            
            network_sample_edgeList = pd.merge(network_sample_edgeList, TP.stack().rename_axis(['lower_level','higher_level']).rename('TP').reset_index(), how='left', on=['lower_level', 'higher_level'])
            network_sample_edgeList = pd.merge(network_sample_edgeList, TN.stack().rename_axis(['lower_level','higher_level']).rename('TN').reset_index(), how='left', on=['lower_level', 'higher_level'])
            network_sample_edgeList = pd.merge(network_sample_edgeList, FN.stack().rename_axis(['lower_level','higher_level']).rename('FN').reset_index(), how='left', on=['lower_level', 'higher_level'])
        
            network_sample_edgeList.loc[network_sample_edgeList.TP == True, 'class'] = "TP"
            network_sample_edgeList.loc[network_sample_edgeList.TN == True, 'class'] = "TN"
            network_sample_edgeList.loc[network_sample_edgeList.FN == True, 'class'] = "FN"
            
            network_sample_edgeList = network_sample_edgeList.drop(["TP", "TN", "FN"], axis=1)
            
            subsamples[frac][rep] = {"Adjacency_Matrix":network_sample,
                                     "Edge_List":network_sample_edgeList}
    
    
    # Add also frac = 1 (Original network)
    network_edgelist['class'] = network_edgelist.apply(lambda row: 'TP' if row['weight'] > 0 else 'TN', axis=1)
    
    subsamples[1][1] = {"Adjacency_Matrix":network,
                          "Edge_List":network_edgelist}
    
    return subsamples

####################################################################################

def export2csv(networks, output_dir):
    
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
                    network.to_csv(output_dir+'/subsamples_edge_lists.csv', index=False, mode='a', header=not pd.io.common.file_exists(output_dir+'/subsamples_edge_lists.csv'))
                    
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
                    
                    meta_data.to_csv(output_dir+'/subsamples_metadata.csv', index=False, mode='a', header=not pd.io.common.file_exists(output_dir+'/subsamples_metadata.csv'))