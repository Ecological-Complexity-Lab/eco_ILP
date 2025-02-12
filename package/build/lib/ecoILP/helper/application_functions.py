import numpy as np
import pandas as pd

from helper.topology_functions import getNetFeatures

def R_features(network, method='rpy2', node=None):

    # if 'subsample_ID' not in network.columns:
    #     network['subsample_ID'] = -1

    if method == 'rpy2':

        import rpy2.robjects as robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter

        ## Convert pandas object to R's dataframe
        with localconverter(robjects.default_converter + pandas2ri.converter):
            network_r = robjects.conversion.py2rpy(network)
        
        ## Save variable in R's environment
        robjects.globalenv['network'] = network_r
        
        ## Call fitSBM function
        r_code = '''
            source("helper/topology_functions.R")
            features = topoFeatures(network)

            # drop columns which are not new features
            drop_columns <- colnames(network) 
            drop_columns <- drop_columns[drop_columns!='link_ID']
            features = features[,!(names(features) %in% drop_columns)]
        '''
        robjects.r(r_code)
        #robjects.r("interaction_probs <- FitSBM(network)")
        
        features_r = robjects.r("as.data.frame(features)")
        
        with localconverter(robjects.default_converter + pandas2ri.converter):
            features = robjects.conversion.rpy2py(features_r)
    
    elif method == 'subprocess':

        import subprocess

        # Save DataFrame to CSV
        input_csv = ".temp/network_input.csv"
        network.to_csv(input_csv, index=False)

        # Define paths
        output_csv = ".temp/network_output.csv"
        remote_input_csv = ".temp/network_input.csv"
        remote_output_csv = ".temp/network_output.csv"

        # Transfer input CSV to the node
        subprocess.run(["scp", input_csv, f"{node}:{remote_input_csv}"], check=True)

        # SSH command to run R script on the node
        ssh_command = f"ssh {node} 'Rscript execute_R_script.R {remote_input_csv} {remote_output_csv}'"
        subprocess.run(ssh_command, shell=True, check=True)

        # Transfer output CSV back to the node
        subprocess.run(["scp", f"{node}:{remote_output_csv}", output_csv], check=True)

        # Read the output CSV into a DataFrame
        features = pd.read_csv(output_csv)

        # Delete local temporary files
        subprocess.run(["rm", input_csv, output_csv], check=True)

    return features

def get_features(edgeList, features_list_py=[], parallel=False):
    
    if 'subsample_ID' not in edgeList.columns:
        edgeList['subsample_ID'] = -1

    # get features (python script)
    features_py = getNetFeatures(edgeList, features_list_py)
    
    # get features (R script)
    features_R = R_features(edgeList) #, method='subprocess', node='bhn27'

    features = pd.merge(features_py, features_R, how="left", left_on='link_ID', right_on='link_ID').replace([np.inf, -np.inf], np.nan)
    
    return features


def get_features_temp2(edgeList):

    # get features (python script)
    features_py = getNetFeatures(edgeList)

    features = features_py.replace([np.inf, -np.inf], np.nan)
    
    # get features (R script)
    # features_R = R_features(edgeList)

    # features = features_R.replace([np.inf, -np.inf], np.nan)
    
    return features

def fill_missing_features(df, model):
    missing_features = set(model.feature_names_in_).difference(df)

    for feature in missing_features:
        df[feature] = 0

    return df