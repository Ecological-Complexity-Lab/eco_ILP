import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, log_loss, brier_score_loss, matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ecoILP import getNetFeatures

# set up logger
# import logging
# logger = logging.getLogger(__name__)
# logging.basicConfig(filename='log.log', level=logging.INFO)


metrics_functions = {
    'f1': lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred), # F1-score
    'precision': lambda y_true, y_pred, y_proba: precision_score(y_true, y_pred, zero_division=0), # Precision
    'recall': lambda y_true, y_pred, y_proba: recall_score(y_true, y_pred), # Recall
    'specificity': lambda y_true, y_pred, y_proba: recall_score(y_true, y_pred, pos_label=0), # Specificity
    'accuracy': lambda y_true, y_pred, y_proba: accuracy_score(y_true, y_pred), # Accuracy
    'roc_auc': lambda y_true, y_pred, y_proba: roc_auc_score(y_true, y_proba), # ROC AUC
    'pr_auc': lambda y_true, y_pred, y_proba: average_precision_score(y_true, y_proba), # Precision-Recall AUC
    'average_precision': lambda y_true, y_pred, y_proba: average_precision_score(y_true, y_proba), # Average Precision
    'f1_macro': lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred, average='macro'), # F1-score macro
    'f1_micro': lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred, average='micro'), # F1-score micro
    'f1_weighted': lambda y_true, y_pred, y_proba: f1_score(y_true, y_pred, average='weighted'), # F1-score weighted
    'log_loss': lambda y_true, y_pred, y_proba: log_loss(y_true, y_proba), # Log Loss
    'brier_score_loss': lambda y_true, y_pred, y_proba: brier_score_loss(y_true, y_proba), # Brier Score Loss
    'mcc': lambda y_true, y_pred, y_proba: matthews_corrcoef(y_true, y_pred), # Matthews Correlation Coefficient
    'informedness': lambda y_true, y_pred, y_proba: recall_score(y_true, y_pred) + recall_score(y_true, y_pred, pos_label=0) - 1, # Youden's informedness
    # 'top_k_accuracy': lambda y_true, y_pred, y_proba: top_k_accuracy_score(y_true, y_pred, k=5) # Top-K Accuracy
}

class ecoILPmodel:

    def __init__(self, model_path=None):

        default_model_path = os.path.join(os.path.dirname(__file__), 'models/ecoILP.joblib')

        self.trained_model = None
        self.model_path = model_path if model_path else default_model_path

    def load_model(self):

        model_data = joblib.load(self.model_path)
        self.trained_model = model_data['trained_model']
    
    def get_model_name(self):
        if self.trained_model.__class__.__name__.endswith('CV'):
            return self.trained_model.estimator.named_steps['classifier'].__class__.__name__

def load_model(model_path=None):
    model = ecoILPmodel(model_path)
    model.load_model()

    return model

def subset_data(df, dataset_link_id=None, cast_target=True):
    
    # Get the subset of the data
    if dataset_link_id is None:
        dataset_link_id = df['link_ID']

    subset_idx = df['link_ID'].isin(dataset_link_id)
    
    # Get the relevant subset of the data
    X_subset = df[subset_idx].iloc[:,:-1]
    y_subset = df[subset_idx].iloc[:,-1]

    # Change 'class' labels to numeric binary
    if cast_target:
        y_subset[y_subset == -1] = 1
    
    return X_subset, y_subset

def sampleLinks(df, fraction=0.2, degree_biased=None, topNodes_col='higher_level', bottomNodes_col='lower_level'):

    adj_matrix = df.pivot(index=bottomNodes_col, columns=topNodes_col, values='weight')

    adj_matrix_np = adj_matrix.to_numpy()
    total_links = int(adj_matrix_np.sum() * fraction)
    
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

    # map between indices and df's link_Id column
    sampled_indices = [link_indices[idx] for idx in sampled_indices]
    sampled_indices = [df.loc[(df[bottomNodes_col] == adj_matrix.index[i]) & (df[topNodes_col] == adj_matrix.columns[j])].index[0] for i, j in sampled_indices]

    return sampled_indices

def handleEdgeList(df, linkID_col='link_ID', topNodes_col='lower_level', bottomNodes_col='higher_level', networkID_col=None, groupID_col=None, weight_col='weight', sample_fraction=None, groundTruth_col = None, community = None, missing_links=None):
    '''
    This function handles the edgelist dataframe by renaming the columns and merging the features dataframe with the edgelist dataframe.
    Previous code was not written in a way that it could be used for other datasets. This function is written to make the code more general.
    '''

    edgelist_df = df.copy()

    # Verify existence of mandatory arguments
    if linkID_col not in edgelist_df.columns:
        print(f'Column "{linkID_col}" not found in edgelist dataframe, using index as link_ID')
        edgelist_df['link_ID'] = edgelist_df.index
    if topNodes_col not in edgelist_df.columns:
        raise ValueError(f'Column "{topNodes_col}" not found in edgelist dataframe')
    if bottomNodes_col not in edgelist_df.columns:
        raise ValueError(f'Column "{bottomNodes_col}" not found in edgelist dataframe')
    if weight_col not in edgelist_df.columns:
        raise ValueError(f'Column "{weight_col}" not found in edgelist dataframe')
    if sample_fraction is not None and sample_fraction <= 0:
        raise ValueError('Sample fraction must be greater than 0')
    if sample_fraction is not None and sample_fraction > 1:
        raise ValueError('Sample fraction must be less than or equal to 1')
    if sample_fraction and groundTruth_col:
        raise ValueError('Cannot sample links and have ground truth at the same time')
    if sample_fraction and missing_links:
        raise ValueError('Cannot sample links and have missing links at the same time')
    if sample_fraction and 'class' in edgelist_df.columns:
        print('Column "class" found in edgelist dataframe, but sample_fraction is not None. Removing column "class"')
        edgelist_df = edgelist_df.drop(columns='class')
    if missing_links and 'class' in edgelist_df.columns:
        print('Column "class" found in edgelist dataframe, but missing_links is not None. Removing column "class"')
        edgelist_df = edgelist_df.drop(columns='class')
    if groundTruth_col is not None and groundTruth_col not in edgelist_df.columns:
        raise ValueError(f'Column "{groundTruth_col}" not found in edgelist dataframe')
    if groundTruth_col is None and 'class' in edgelist_df.columns:
        print('Column "class" found in edgelist dataframe, but groundTruth_col is None. Removing column "class"')
        edgelist_df = edgelist_df.drop(columns='class')

    # Rename columns to fit the code
    if networkID_col is not None:
        edgelist_df = edgelist_df.rename(columns={networkID_col: 'name'}) # TODO: 'networkID' is better than 'name
    else:
        edgelist_df['name'] = 'network_1'
    if groupID_col is not None:
        edgelist_df = edgelist_df.rename(columns={groupID_col: 'community'})# TODO: 'groupID' is better than 'community'
    else:
        if community is None:
            community = 'community_1'
        edgelist_df['community'] = community
    if linkID_col != 'link_ID':
        edgelist_df = edgelist_df.rename(columns={linkID_col: 'link_ID'})
    if bottomNodes_col != 'lower_level':
        edgelist_df = edgelist_df.rename(columns={bottomNodes_col: 'lower_level'})
    if topNodes_col != 'higher_level':
        edgelist_df = edgelist_df.rename(columns={topNodes_col: 'higher_level'})
    if weight_col != 'weight':
        edgelist_df = edgelist_df.rename(columns={weight_col: 'weight'})

    # Convert the weight column to binary
    edgelist_df['weight'] = (edgelist_df['weight'] > 0).astype('int')

    if sample_fraction is not None:
        missing_links = sampleLinks(edgelist_df, fraction=sample_fraction, degree_biased=None)

    # "Hide" links if needed (i.e. set forcing missing links)
    if missing_links is not None:

        # Save the true values of edges
        edgelist_df['class'] = edgelist_df['weight']

        # Change the weight of the new missing links to 0
        missing_links_mask = edgelist_df['link_ID'].isin(missing_links)

        # Set the weight of the missing links to 0
        edgelist_df.loc[missing_links_mask, 'weight'] = 0

        # Set the class of the missing links to -1
        edgelist_df.loc[missing_links_mask, 'class'] = -1

    
    return edgelist_df

def extractFeatures(edgelist_df, return_features_only=False):
    
    features_list = [
        'network_size', 'species_ratio', 'interactions_count', 'edge_connectivity', 'bipartite_clustering', 'Spectral_bipartivity',
        'average_clustering', 'degree_assortativity_coefficient', 'global_efficiency', 'local_efficiency', 'connected_components', 
        'degree', 'latapy_clustering', 'node_redundancy', 'betweenness_centrality', 'degree_centrality', 'closeness_centrality', 
        'average_neighbor_degree', 'pagerank', 'hits_hubs', 'hits_authorities', 'preferential_attachment', 'shortest_path_length', 
        'shortest_paths_count', 'friends_measure',
    ]

    features_df = getNetFeatures(edgelist_df[['link_ID', 'lower_level', 'higher_level', 'name', 'community', 'weight']], features_list)

    if return_features_only:
        return features_df

    # Merge features dataframe with edgelist dataframe
    edgelist_df = edgelist_df.reset_index().merge(features_df, on='link_ID', how='left').set_index('index') # preserving the index

    # Make sure groundTruth column is last
    if 'class' in edgelist_df.columns:
        edgelist_df = edgelist_df[[col for col in edgelist_df.columns if col != 'class'] + ['class']]
    
    return edgelist_df

def predictLinks(df, model, threshold=0.5, return_pred=True, return_proba=True):

    X = df.copy()

    if 'class' in df.columns:
        X, _ = subset_data(df, cast_target=True)

    # Get probabilities
    y_proba = model.trained_model.predict_proba(X)[:,1]

    # Get predictions based on probabilities, using the given threshold
    y_pred = (y_proba >= threshold).astype('int')

    if return_pred and return_proba:
        return y_proba, y_pred
    elif return_proba:
        return y_proba
    else:
        return y_pred

def plotMetrics(df, probabilities, threshold=0.5, plots=['confusion_matrix', 'grouped_evaluation', 'roc_curve', 'pr_curve', 'probs_distribution'], model_name='EcoILP'):
    '''
    This function plots the results of the predictions.
    '''

    # Drop existing links column
    mask = df['weight'] == 1
    df = df[~mask]
    probabilities = probabilities[~mask]

    if 'class' in df.columns:
        X, y_true = subset_data(df, cast_target=True)
    else:
        raise ValueError('Column "class" not found in the dataframe')
    
    return multi_plot(
        X, 
        y_true, 
        y_proba=probabilities,
        threshold=threshold, 
        plots=plots,
        model_name = model_name,
        show=False
    )

def plotProbsMatrix(df, probabilities, threshold=0.5, topNodes_col='higher_level', bottomNodes_col='lower_level', figsize=(10, 10)):
    """
    Plot probability matrix with class indicators.
    Args:
        df: DataFrame with columns [bottomNodes_col, topNodes_col, weight, class]
        probabilities: Array of prediction probabilities
        threshold: Classification threshold
        topNodes_col: Column name for top nodes
        bottomNodes_col: Column name for bottom nodes
    """

    # Clean data - remove quotes and create fresh copy
    df = df.copy().reset_index(drop=True)

    # Combine probabilities with edge list
    df['prediction'] = probabilities
    
    # Ignore predictions for existing links
    df['prediction'] = df.apply(lambda row: 1 if row['weight'] == 1 else row['prediction'], axis=1)
    
    # convert edge list to matrix
    matrix = df.pivot(index=bottomNodes_col, columns=topNodes_col, values='prediction')

    # Plot setup
    # fig, ax = plt.subplots(figsize=figsize)
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    # Plot heatmap
    sns.heatmap(
        matrix, 
        cmap=plt.cm.Blues,
        vmin=0,
        vmax=1,
        cbar=True,
        annot=True,
        fmt='.2f',
        square=True,
        ax=ax,
        alpha=0.8,
        linewidths=2,
        cbar_kws={'shrink': 0.3}  # Make colorbar shorter
    )

    colors = {
        'correct': '#00BF7D',
        'wrong': "#F8766D"
    }

    # Map bottom and top nodes to matrix indices
    bottom_node_to_y = {node: idx for idx, node in enumerate(matrix.index)}
    top_node_to_x = {node: idx for idx, node in enumerate(matrix.columns)}

    # Adjust rectangles for linewidth padding
    padding = 0.05

    # Add class indicators if available
    if 'class' in df.columns:
        for idx, row in df.iterrows():
            bottom = row[bottomNodes_col]
            top = row[topNodes_col]

            # Get matrix coordinates using the mappings
            y = bottom_node_to_y.get(bottom)
            x = top_node_to_x.get(top)
            pred = row['prediction']

            # Ensure coordinates are valid
            if y is not None and x is not None:
                # Draw rectangles for class indicators
                if row['class'] in [0, -1]:
                    color = colors['correct'] if (row['class'] == 0 and pred < threshold) or(row['class'] == -1 and pred >= threshold)  else colors['wrong']
                    rect = patches.Rectangle(
                        (x + padding, y + padding),
                        1 - 2 * padding,
                        1 - 2 * padding,
                        fill=False,
                        edgecolor=color,
                        linewidth=2
                    )
                    ax.add_patch(rect)

                    # Add X for negative classes
                    if row['class'] == -1:
                        ax.text(
                            x + 0.8, y + 0.22,
                            'X',
                            color=colors['wrong'],
                            ha='center',
                            va='center',
                            fontweight='bold',
                            alpha=0.8
                        )
    
    
    # plt.title('Probability Matrix')
    # plt.tight_layout()
    # plt.show()

    # return the plot
    return fig

def evaluate(y_true=None, y_proba=None, threshold=0.5, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss']):

    # Get predictions based on probabilities, using the given threshold
    y_pred = (y_proba >= threshold).astype('int') 
    
    results = {}

    for metric in metrices:
        if metric in metrics_functions:
            fun = metrics_functions[metric]
            results[metric] = fun(y_true, y_pred, y_proba)
        else:
            print('Unknown metric: ' + metric)           

    # Convert scalar values to single-item lists
    results = {k: [v] for k, v in results.items()}
    
    return pd.DataFrame.from_dict(results)#.round({'f1_score': 2, 'precision': 2,  'recall': 2, 'accuracy': 2, 'ROC_AUC': 2, 'PR_AUC': 2, 'average_precision': 2, 'f1_macro': 2, 'f1_micro': 2, 'f1_weighted': 2 })

def plot_confusion_matrix(y_true=None, y_proba=None, threshold=0.5, normalize=True, ax=None, title='Confusion Matrix', show=True):

    # Get predictions based on probabilities, using the given threshold
    y_pred = (y_proba >= threshold).astype('int') 

    # Deal with normalization
    if normalize:
        cm_normalize = 'true'
    else:
        cm_normalize = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize=cm_normalize)

    # Create figure
    if ax is None: # if ax is not a part of a subplot
        fig, ax = plt.subplots(figsize=(8, 8))
    cm_disp = ConfusionMatrixDisplay(cm, display_labels=['non-existing\nlinks (0)','missing\nlinks (1)'])
    cm_disp.plot(ax=ax, cmap = plt.cm.Blues, include_values=True)
    #cm_disp.ax_.grid()
    cm_disp.ax_.set(xlabel='Predicted Class', ylabel='True Class')
    cm_disp.ax_.set_title(title)
    
    # Fix colorbar scale
    cm_disp.im_.colorbar.remove()
    divider = make_axes_locatable(ax) 
    cax = divider.append_axes("right", size="5%", pad=0.05)
    colorbar = ax.figure.colorbar(cm_disp.im_, ax=ax, cax=cax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    colorbar.mappable.set_clim(0, 1)  # Set the color bar limits to 0 and 1

    # Plot
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()
    return None

def plot_single_evaluation(y_true=None, y_proba=None, threshold=0.5, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss'], ax=None, title='Evaluation Metrics', show=True):
        
        results = evaluate(y_true, y_proba, threshold=threshold, metrices=metrices)

        # convert to wide format
        results = results.T.reset_index().rename(columns={'index': 'metric', 0: 'value'})

        # Create figure
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))


        # same as above but with sns
        sns.barplot(x='metric', y='value', data=results, ax=ax).set_title(title)

        # Plot
        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

        return results

def grouped_evaluation(y_true=None, y_proba=None, groups=None, threshold=0.5, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss']):

    # Get predictions based on probabilities, using the given threshold
    y_pred = (y_proba >= threshold).astype('int') 

    # Combine y_pred, y_proba, y_test and groups into one dataframe
    y_combined = pd.concat([ 
        pd.DataFrame(y_pred).rename(columns={0:'y_pred'}), 
        pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), 
        pd.DataFrame(y_true).reset_index(drop=True).rename(columns={'class':'y_true'}),
        pd.DataFrame(groups).rename(columns={0:'group'})], axis=1)
    
    # Group by group_by
    groups = y_combined.groupby('group') 
    
    # Calculate metrics for each group
    results_by_groups = pd.concat([
        groups.apply(lambda group: metrics_functions[metric](group.y_true, group.y_pred, group.y_proba)).rename(metric)
        for metric in metrices
    ], axis=1).reset_index().rename(columns={'index': 'group'}).round({metric: 2 for metric in metrices})
    
    return results_by_groups

def plot_grouped_evaluation(X=None, y_true=None, y_proba=None, group_by = 'name', threshold=0.5, split_by=None, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss'], ax=None, title='Boxplots of metrices per network', show=False):

    if group_by in X.columns:
        groups = np.array(X[group_by].values)
    else:
        raise ValueError(f'Column "{group_by}" not found in the dataframe')

    results_by_groups = grouped_evaluation(y_true, y_proba, groups=groups, threshold=threshold, metrices=metrices)

    # Create figure
    if ax is None: # if ax is not a part of a subplot
        fig, ax = plt.subplots(figsize=(15, 10))

    if split_by is not None:
        results_by_groups = results_by_groups.merge(X[[split_by, group_by]].rename(columns={group_by: 'group'}).drop_duplicates(['group']), how="left", left_on='group', right_on='group')
        # g = results_by_groups.boxplot(ax=ax, by=split_by) #, by=split_by
        df_plot = results_by_groups.reset_index().melt(id_vars=['group', split_by], value_vars=results_by_groups.columns)
        sns.boxplot(x='variable', y='value', hue=split_by, data=df_plot, ax=ax)
        ax.set_title(title)
    else:
        g = results_by_groups.boxplot(ax=ax)
        g.set_title(title)
        plt.sca(ax) # set the current axes for the pyplot state machine | make possible for using matplotlib.pyplot methods
    
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45, ha='right')
    # ax.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)  # Limit the y-axis from 0 to 1

    # Plot
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    return results_by_groups

def plot_roc_curve(X=None, y_true=None, y_proba=None, split_by=None, threshold=0.5, ax=None, title='ROC Curve', model_name='Model', return_curves=False, show=True):
    
    if split_by is not None:

        plots = []

        # Get groups for test data
        groups_test = np.array(X[split_by].values) 

        # Combine y_pred, y_proba, y_true and groups_test into one dataframe
        y_combined = pd.concat([ 
            pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), 
            pd.DataFrame(y_true).reset_index(drop=True).rename(columns={'class':'y_true'}),
            X['community'].reset_index(drop=True),
            pd.DataFrame(groups_test).rename(columns={0:'group'})], axis=1)
        
        # Group by group_by
        groups = y_combined.groupby('community') 

        for community_name, group in groups:

            # ROC-AUC
            fpr, tpr, thresholds_roc = roc_curve(group.y_true, group.y_proba)
            roc_auc = auc(fpr, tpr)

            # ROC Curve
            rc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=community_name)
            plots.append(rc_disp)

        if return_curves:
            return plots
        
        # Create figure
        if ax is None: # if ax is not a part of a subplot
            fig, ax = plt.subplots(figsize=(15, 10))

        ## ROC-AUC Curve plot
        for p in plots:
            p.plot(ax=ax)

        ax.grid()
        ax.legend()
        ax.plot([0,1], [0,1], ls = ':')
        ax.set_title(title)

    else:
        # ROC-AUC
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        ## ROC-AUC Curve plot
        rc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
        rc_disp.plot(ax=ax)
        rc_disp.ax_.grid()
        rc_disp.ax_.legend()
        rc_disp.ax_.plot([0,1], [0,1], ls = ':')
        rc_disp.ax_.set_title(title)

        idx = np.argmin(np.abs(thresholds_roc - threshold))
        rc_disp.ax_.annotate(f'Threshold: {threshold}', (fpr[idx], tpr[idx]), xytext=(fpr[idx] + 0.05, tpr[idx] - 0.05),
                            arrowprops=dict(facecolor='black', arrowstyle='->'))

    # Plot
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    return None

def plot_pr_curve(X=None, y_true=None, y_proba=None, split_by=None, threshold=0.5, ax=None, title='Precision-Recall Curve', model_name='Model', show=True):

    # Create figure
    if ax is None: # if ax is not a part of a subplot
        fig, ax = plt.subplots(figsize=(15, 10))

    if split_by is not None:

        plots = []

        # Get groups for test data
        groups_test = np.array(X[split_by].values) 

        # Combine y_pred, y_proba, y_test and groups_test into one dataframe
        y_combined = pd.concat([ 
            pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), 
            pd.DataFrame(y_true).reset_index(drop=True).rename(columns={'class':'y_true'}),
            X['community'].reset_index(drop=True),
            pd.DataFrame(groups_test).rename(columns={0:'group'})], axis=1)
        
        # Group by group_by
        groups = y_combined.groupby('community') 

        for community_name, group in groups:

            # Precision-Recall
            precision, recall, thresholds_pr = precision_recall_curve(group.y_true, group.y_proba)
            pr_auc = average_precision_score(group.y_true, group.y_proba)

            # PR Curve
            pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=pr_auc, estimator_name=community_name)
            plots.append(pr_disp)

        ## Precision-Recall Curve plot
        for p in plots:
            p.plot(ax=ax)
        #pr_disp.ax_.legend()
        no_skill = len(y_true[y_true==1]) / len(y_true)
        ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        ax.grid()
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.legend()
        ax.set_title(title)
    else:
        # Precision-Recall
        precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
        average_precision = average_precision_score(y_true, y_proba)

        ## Precision-Recall Curve plot
        pr_disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        pr_disp.plot(ax=ax, label=model_name+' (area = %0.2f)'% average_precision)
        no_skill = len(y_true[y_true==1]) / len(y_true)
        pr_disp.ax_.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
        pr_disp.ax_.grid()
        pr_disp.ax_.set_xlim([0.0, 1.0])
        pr_disp.ax_.set_ylim([0.0, 1.0])
        pr_disp.ax_.legend()
        pr_disp.ax_.set_title(title)

    # Plot
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()
    return None

def plot_pr_vs_threshold(X=None, y_true=None, y_proba=None, threshold=0.5, group_by=None, ax=None, title='Precision-Recall vs Threshold', show=True):

    if group_by is None:

        # Precision-Recall
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

        if ax is None: # if ax is not a part of a subplot
            fig, ax = plt.subplots(figsize=(6,6))

        ax.plot(thresholds, precision[:-1], 'blue', label='Precisions')
        ax.plot(thresholds, recall[:-1], 'green', label='Recalls')
        ax.set_ylabel('Level of Precision and Recall', fontsize=9)
        ax.set_title(title)
        ax.set_xlabel('Thresholds', fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.set_ylim([0,1])
        ax.axvline(x=0.5, linewidth=3, color='#0B3861')

    else:
        # Get predictions based on probabilities, using the given threshold
        y_pred = (y_proba >= threshold).astype('int') 

        # Get groups for test data
        groups_test = np.array(X[group_by].values) 

        # Combine y_pred, y_proba, y_true and groups_test into one dataframe
        y_combined = pd.concat([ 
            pd.DataFrame(y_pred).rename(columns={0:'y_pred'}), 
            pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), 
            pd.DataFrame(y_true).reset_index(drop=True).rename(columns={'class':'y_true'}),
            pd.DataFrame(groups_test).rename(columns={0:'group'})], axis=1)
        
        # Group by group_feature
        groups = y_combined.groupby('group') 

        # Precision-Recall for each group
        pr_by_groups = [precision_recall_curve(group.y_true, group.y_proba) for name, group in groups] # Precision-Recall

        thresholds4plot = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        pr_df = pd.DataFrame()

        for grouped_result in pr_by_groups:

            # Precision-Recall
            precision, recall, thresholds = grouped_result

            idx =  [(np.abs(thresholds - fixed_threshold)).argmin() for fixed_threshold in thresholds4plot] # find (indices of) values nearest desired thresholds

            pr_df = pd.concat([pr_df, pd.DataFrame({'metric':['precision']*len(thresholds4plot), 'threshold':thresholds4plot, 'score':precision[idx]})], ignore_index=True)
            pr_df = pd.concat([pr_df, pd.DataFrame({'metric':['recall']*len(thresholds4plot), 'threshold':thresholds4plot, 'score':recall[idx]})], ignore_index=True)

        # Create figure
        if ax is None: # if ax is not a part of a subplot
            fig, ax = plt.subplots(figsize=(15, 10))
        sns.boxplot(x='threshold', y='score', hue='metric', data=pr_df, palette='Blues', ax=ax).set(title='Precision and Recall Scores\nas a function of the decision threshold\ndistribution by groups')
        ax.axvline(0.5)

    # Plot
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    return None

def plot_probs_distribution(y_true=None, y_proba=None, axes=None, show=True):

    y_new = pd.concat([pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), pd.DataFrame(y_true).reset_index(drop=True)], axis=1)
    
    if axes is None: # if ax is not a part of a subplot
        fig, axes = plt.subplots(1,2, figsize=(10,4))

    cm = sns.color_palette("RdBu", 20, as_cmap=True)
    cm_r = sns.color_palette("RdBu_r", 20, as_cmap=True)

    ax=axes[0]
    sns.histplot(data=y_new[y_new['class'] == 1], x="y_proba", kde=True, ax=ax).set(title='Missing link probability distribution') 
    ax.axvline(0.5)

    for bin_ in ax.patches:
        bin_midpoint = bin_.get_x()
        bin_.set_facecolor(cm(bin_midpoint))
    
    
    ax=axes[1]
    sns.histplot(data=y_new[y_new['class'] == 0], x="y_proba", kde=True, ax=ax).set(title='Non-existing links probability distribution')
    ax.axvline(0.5)

    for bin_ in ax.patches:
        bin_midpoint = bin_.get_x()
        bin_.set_facecolor(cm_r(bin_midpoint))
    
    # Plot
    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    return None

def multi_plot(X=None, y_true=None, y_proba=None, threshold=0.5, group_by='name', model_name = 'model', plots=[], show=True):

    if y_proba is None:
        raise ValueError('y_proba is required for plotting')
    
    # Some plots require more than one subplot
    extra_subplots = 0
    if 'probs_distribution' in plots:
        extra_subplots += 1
    
    required_subplots = len(plots) + extra_subplots
    nrows= -(required_subplots // -4)
    fig, axes = plt.subplots(figsize = (24,6*nrows), ncols = min(required_subplots, 4), nrows = nrows)#, layout="constrained")
    
    ax_iter = iter(axes.flat)

    for plot in plots: 
        ax = next(ax_iter) # Get next axes object
        if plot == 'confusion_matrix':
            plot_confusion_matrix(y_true, y_proba, threshold=threshold, show=False, ax=ax)
        elif plot == 'single_evaluation':
            plot_single_evaluation(y_true, y_proba, threshold=threshold, metrices=['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall', 'mcc'], show=False, ax=ax)
        elif plot == 'grouped_evaluation':
            plot_grouped_evaluation(X, y_true, y_proba, threshold=threshold, group_by = group_by, metrices=['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall', 'mcc'], show=False, ax=ax)
        elif plot == 'grouped_evaluation_split':
            plot_grouped_evaluation(X, y_true, y_proba, split_by='community', threshold=threshold, group_by = group_by, metrices=['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall', 'mcc'], show=False, ax=ax)
        elif plot == 'roc_curve':
            plot_roc_curve(X, y_true, y_proba, threshold=threshold, show=False, ax=ax)
        elif plot == 'pr_curve':
            plot_pr_curve(X, y_true, y_proba, threshold=threshold, show=False, ax=ax, model_name=model_name)
        elif plot == 'roc_curve_split':
            plot_roc_curve(X, y_true, y_proba, split_by='community', threshold=threshold, show=False, ax=ax)
        elif plot == 'pr_curve_split':
            plot_pr_curve(X, y_true, y_proba, split_by='community', threshold=threshold, show=False, ax=ax, model_name=model_name)
        elif plot == 'pr_curve_vs_threshold':
            plot_pr_vs_threshold(X, y_true, y_proba, threshold=threshold, show=False, ax=ax)
        elif plot == 'pr_curve_vs_threshold_grouped':
            plot_pr_vs_threshold(X, y_true, y_proba, threshold=threshold, group_by=group_by, show=False, ax=ax)
        elif plot == 'probs_distribution':
            plot_probs_distribution(y_true, y_proba, show=False, axes=[ax, next(ax_iter)])

    if show:
        plt.tight_layout()
        plt.show()
        plt.close()

    # return the figure
    return fig


# ----------------- Interactive Interface -----------------

import gradio as gr

def create_probability_styler(df, original_values=None):
    """Create a styled dataframe for probability display"""
    # Create a copy to avoid modifying the original
    styled_df = df.copy()

    # Create the styler
    def color_cells(val, original_val):
        try:
            # Define a string variable for the cell style

            if val >= 1:  # Link
                style = 'background-color: rgba(7, 79, 143, 1.0); color: black;'
            elif 0 < val < 1:  # Probability values
                style = f'background-color: rgba(7, 79, 143, {val}); color: black;'
            else:  # Zero or empty cells
                style = 'background-color: white;'

            if original_values is not None:
                if original_val == 1:
                    style = 'background-color: rgba(176, 176, 176, 1.0); color: black;'
                elif original_val == 0:
                    if val < 0.5:
                        style = f'background-color: rgba(2, 173, 93, {val}); color: black;'
                    else:
                        style = f'background-color: rgba(219, 57, 57, {val}); color: black;'
                elif original_val == -1:
                    if val >= 0.5:
                        style = f'background-color: rgba(2, 173, 93, {val}); border: 2px dashed green; color: black;'
                    else:
                        style = f'background-color: rgba(219, 57, 57, {val}); border: 2px dashed red; color: black;'
            
            return style
        
        except (ValueError, TypeError):
            return 'background-color: white'  # For non-numeric values (like row labels)
    
    # Apply styling
    if original_values is not None:
        styler = styled_df.style.apply(lambda x: [color_cells(v, ov) for v, ov in zip(x, original_values.loc[x.name])], axis=1, subset=styled_df.columns[1:])
        # styler = styled_df.style.applymap(lambda val: color_cells(val, -1), subset=styled_df.columns[1:])
    else:
        # styler = styled_df.style.applymap(color_cells, subset=styled_df.columns[1:])
        styler = styled_df.style.applymap(lambda val: color_cells(val, None), subset=styled_df.columns[1:])
    
    return styler

def update_matrix_from_file(csv_file, file_type):
    if csv_file is not None:
        if file_type == 'Adjacency Matrix':
            matrix_df = pd.read_csv(csv_file.name, index_col=0)
            matrix_df.index.name = None
            top_nodes = matrix_df.columns.tolist()
            matrix_df = matrix_df.reset_index()
            matrix_df.columns = [""] + top_nodes
            # Convert values to numeric, coerce errors to NaN
            matrix_df.iloc[:, 1:] = matrix_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_styler = create_probability_styler(matrix_df)
            input_html = input_styler.to_html()
            return gr.update(value=input_styler), gr.update(value=input_html), gr.update(value=None)
            
        elif file_type == 'Edge List':
            df = pd.read_csv(csv_file.name)
            pivot_table = df.pivot(index='lower_level', columns='higher_level', values='weight').fillna(0)
            pivot_table.index.name = None
            top_nodes = pivot_table.columns.tolist()
            pivot_table = pivot_table.reset_index()
            pivot_table.columns = [""] + top_nodes
            # Convert values to numeric
            pivot_table.iloc[:, 1:] = pivot_table.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)
            input_styler = create_probability_styler(pivot_table)
            input_html = input_styler.to_html()
            return gr.update(value=pivot_table), gr.update(value=input_html), gr.update(value=None)
    else:
        return gr.update(), gr.update(), gr.update()

def predict_network(matrix_values, slider_value=0, community_value='None'):

    matrix_df = matrix_values
    top_nodes = matrix_df.columns[1:].tolist()
    bottom_nodes = matrix_df.iloc[:, 0].tolist()
    data_list = []
    link_id = 0

    # Convert values to numeric, coerce errors to NaN
    matrix_df.iloc[:, 1:] = matrix_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0)

    # Create initial dataframe
    for i, bottom_node in enumerate(bottom_nodes):
        for j, top_node in enumerate(top_nodes):
            try:
                weight = float(matrix_df.iloc[i, j+1])
                # Convert any positive value to 1, keep 0 as 0
                weight = 1.0 if weight > 0 else 0.0
            except ValueError:
                return gr.update(value="Error: Matrix contains non-numeric values."), gr.update(value=None), None
            data_list.append({
                'link_ID': link_id,
                'higher_level': top_node,
                'lower_level': bottom_node,
                'weight': weight
            })
            link_id += 1

    df = pd.DataFrame(data_list)

    model = load_model() # Load the model (default model)
    
    # Run the prediction pipeline
    try:

        sample_fraction = slider_value / 100 if slider_value > 0 else None
        
        dataframe = handleEdgeList(
            df,
            linkID_col='link_ID',
            topNodes_col='higher_level',
            bottomNodes_col='lower_level',
            networkID_col=None,
            groupID_col=None,
            weight_col='weight',
            community=community_value,
            sample_fraction=sample_fraction
        )

        dataframe_with_features = extractFeatures(dataframe)
        probabilities, classifications = predictLinks(dataframe_with_features, model)
    
        # Reshape predictions into matrix form
        prediction_matrix = probabilities.reshape(len(bottom_nodes), len(top_nodes))
        
        # Create output dataframe with the same structure as input
        pivot_values = 'class' if 'class' in dataframe.columns else 'weight'
        result_df = dataframe.pivot(index='lower_level', columns='higher_level', values=pivot_values).fillna(0)
        result_df.index.name = ""
        result_df = result_df.reset_index()
        
        # Get numerical values (excluding first column which has labels)
        values = result_df.iloc[:, 1:].astype(float)

        # Save original values before replacing them
        original_values = values.copy() if slider_value > 0 else None

        # Create final matrix:
        # - where input was >= 1, put 1
        # - where input was 0, put the prediction probability
        new_values = np.where(values > 0, 1, prediction_matrix)
        
        # Update the dataframe with new values
        result_df.iloc[:, 1:] = new_values
        
        # Apply styling
        probability_styler = create_probability_styler(result_df, original_values).format(precision=2)
        probability_html = probability_styler.hide(axis="index").to_html()
                
        # Initialize metrics_fig and probs_matrix_fig
        metrics_fig = None
        probs_matrix_fig = None
        
        # Generate metrics plots if slider_value is not 0
        if slider_value > 0:
            metrics_fig = plotMetrics(
                dataframe_with_features, 
                probabilities, 
                plots=['confusion_matrix', 'single_evaluation', 'roc_curve', 'pr_curve', 'probs_distribution']
            )
            probs_matrix_fig = plotProbsMatrix(dataframe, probabilities, figsize=(14,8))
        
        return gr.update(value=probability_html), gr.update(value=probability_styler), metrics_fig, probs_matrix_fig
        
    except Exception as e:
        return gr.update(value=f"Error during prediction: {e}"), gr.update(value=None), None, None



def create_gradio_interface():
    # Initialize headers and values
    initial_top_nodes = [f'Top_{i}' for i in range(5)]
    # initial_bottom_nodes = [f'Bottom_{i}' for i in range(5)]
    initial_df = pd.DataFrame([[f'Bottom_{i}'] + [0]*len(initial_top_nodes) for i in range(5)],
                            columns=[""] + initial_top_nodes)
    initial_styler = create_probability_styler(initial_df)
    initial_html = initial_styler.to_html()

    css = """
        .dataframe-container {
            overflow-x: auto !important;
            max-width: 100%;
        }
        /* First column (index) */
        .dataframe-container table th:first-child,
        .dataframe-container table td:first-child {
            min-width: 100px !important;
            max-width: 150px !important;
        }
        /* Data columns */
        .dataframe-container table th:not(:first-child),
        .dataframe-container table td:not(:first-child) {
            min-width: 50px !important;
            max-width: 80px !important;
            text-align: center;
        }
    """

    with gr.Blocks(css=css) as demo:
        gr.Markdown("# Network Link Prediction")
        
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    pass
            with gr.Column(variant='panel', scale=1):
                file_type = gr.Dropdown(
                    label="Select File Type", 
                    choices=['Adjacency Matrix', 'Edge List'], 
                    value='Adjacency Matrix',
                )
                csv_input = gr.File(
                    label="Upload CSV File", 
                    file_types=['.csv'], 
                    file_count='single', 
                    # container=False,
                    # height='50px'
                )
        
        with gr.Row():            
            with gr.Column(variant='panel'):
                community = gr.Dropdown(
                    label="Community",
                    choices=['Plant-Pollinator', 'Plant-Herbivore', 'Host-Parasite', 'Plant-Seed Dispersers', 'Other'],
                    value='Plant-Pollinator',
                    # info="Select the community to use for link prediction."
                )
                with gr.Tabs():
                    with gr.Tab("Adjacency Matrix"):
                        matrix_input = gr.Dataframe(
                            value=initial_df,
                            label="Input Matrix",
                            interactive=True,
                            wrap=False,
                            elem_classes="dataframe-container"
                        )
                    with gr.Tab("Edge List"):
                        pass
        
            with gr.Column(variant='panel'):
                slider = gr.Slider(
                    minimum=0, 
                    maximum=100, 
                    step=5, 
                    value=0, 
                    label="Existing links removal percentage",
                    info=r"Choose between 0% and 100%. This will remove a percentage of existing links from the input matrix.",
                )
                with gr.Tabs():
                    with gr.Tab("DataFrame"):
                        probability_matrix_output_df = gr.Dataframe(
                            value=initial_df,
                            label="Probability Matrix",
                            interactive=False,
                            wrap=False,
                            elem_classes="dataframe-container"
                        )
                    with gr.Tab("HTML"):
                        probability_matrix_output_html = gr.HTML(
                            value=initial_html, 
                            label="Probability Matrix",
                            elem_classes="dataframe-container"
                        )
        
        with gr.Row():
            with gr.Column():
                pass
            predict_button = gr.Button("Predict")
            auto_update_state = gr.Checkbox(
                label="Auto Update", 
                info="Automatically update the output when input changes.",
                value=True,
                interactive=True,
            )
        
        with gr.Column(variant='panel'):
            output_plot_metrics = gr.Plot(label="Metrics")
            output_plot_probs_matrix = gr.Plot(label="Probability Matrix Plot")

        # Connect components
        csv_input.change(
            update_matrix_from_file,
            inputs=[csv_input, file_type],
            outputs=[matrix_input, probability_matrix_output_html, probability_matrix_output_df]
        )
        predict_button.click(
            predict_network,
            inputs=[matrix_input, slider, community],
            outputs=[probability_matrix_output_html, probability_matrix_output_df, output_plot_metrics, output_plot_probs_matrix]
        )
        
        @gr.on(inputs=[matrix_input, slider, community, auto_update_state],outputs=[probability_matrix_output_html, probability_matrix_output_df, output_plot_metrics, output_plot_probs_matrix])
        def predict_network_caller(x1, x2, x3, auto_update_state):
            if auto_update_state:
                return predict_network(x1, x2, x3)
            else:
                return gr.skip(), gr.skip(), gr.skip(), gr.skip()

    return demo

