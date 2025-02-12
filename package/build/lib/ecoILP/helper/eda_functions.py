import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

####################################################################################
from helper.debugger_functions import setGlobal, stop

def get_df_by_lvl(df, features_lvl, node_trophic):

    # Set a dataframe for each level
    df_subsample_unique = df.drop_duplicates(subset = ["subsample_ID"]) # Get a single sample per network (for network-level analysis)

    df_node_lvl = df[df.columns.intersection(features_lvl['node']+['subsample_ID', 'fraction', 'class'])] # problematic

    higher_trophic_features = [feature for feature,trophic_lvl in node_trophic.items() if trophic_lvl == 'higher']
    lower_trophic_features = [feature for feature,trophic_lvl in node_trophic.items() if trophic_lvl == 'lower']

    df_by_lvl = {
        'network':df_subsample_unique[df_subsample_unique.columns.intersection(features_lvl['network']+['class'])],
        'node':df_node_lvl,
        'node_higher':df_node_lvl[list(df_node_lvl.columns.intersection(higher_trophic_features))+['subsample_ID', 'fraction', 'class', 'higher_level']].drop_duplicates(subset = ["higher_level", 'subsample_ID']),
        'node_lower':df_node_lvl[list(df_node_lvl.columns.intersection(lower_trophic_features))+['subsample_ID', 'fraction', 'class', 'lower_level']].drop_duplicates(subset = ["lower_level", 'subsample_ID']),
        'link':df[df.columns.intersection(features_lvl['link']+['subsample_ID', 'fraction', 'class'])],
        'all':df
    }
    
    return df_by_lvl

####################################################################################

# from helper.machine_learning_functions import set_transformer
# def transform(df):

#     irrelevant_features = ['name', 'subsample_ID', 'fraction', 'class', 'link_ID', 'lower_level', 'higher_level']
#     irrelevant_features = [feature for feature in irrelevant_features if feature in df.columns]
#     df_transformed = df[df.columns.difference(irrelevant_features)]

#     preprocessor = set_transformer(df_transformed[df_transformed.columns.difference(irrelevant_features)])
#     df_transformed = pd.DataFrame(preprocessor.fit_transform(df_transformed), columns=preprocessor.get_feature_names_out())

#     df_transformed = pd.merge(df[irrelevant_features], df_transformed, how='right', right_index=True, left_index=True)

#     return df_transformed

####################################################################################

def plot_categoric(df, cols = [], title = ""):
    
    plt.style.use('default')
    
    for col in cols:
        fig, ax = plt.subplots(figsize=(3, 5))
        df[col].value_counts().plot(x=col, kind='bar', rot=45, title=col+' '+title)

        # add count labels
        for bar in ax.patches:
          # The text annotation for each bar should be its height.
          bar_value = bar.get_height()
          # Format the text with commas to separate thousands. You can do
          # any type of formatting here though.
          text = f'{bar_value:,}'
          # This will give the middle of each bar on the x-axis.
          text_x = bar.get_x() + bar.get_width() / 2
          # get_y() is where the bar starts so we add the height to it.
          text_y = bar.get_y() + bar_value
          # If you want a consistent color, you can just set it as a constant, e.g. #222222
          ax.text(text_x, text_y, text, ha='center', va='bottom', color=bar.get_facecolor(), size=12)
            
        plt.tight_layout()
        plt.show()

####################################################################################

def filter_dataframe(df_by_lvl, level='all', fractions=None, communities=None):

    df = df_by_lvl[level]
    
    # Create a base boolean mask
    bool_mask = pd.Series(True, index=df.index)

    if fractions:
        # if fraction columns is not avialable | delete this
        # subsamples_by_fraction = meta.groupby('fraction')['subsample_ID'].apply(list).to_dict()
        # relevant_subsamples = [value for key,values in subsamples_by_fraction.items() if key in fractions for value in values]
        # bool_mask = bool_mask & df['subsample_ID'].isin(relevant_subsamples)

        bool_mask = bool_mask & df['fraction'].isin(fractions)

    return df[bool_mask]

def plot_feature_distributions(df_by_lvl, lvl):

    # Get a list of features names to plot
    irrelevant_features = ['name', 'subsample_ID', 'community', 'fraction', 'symbiotic_relationship', 'class', 'link_ID', 'lower_level', 'higher_level']

    fractions = [0.7, 0.8, 1] # later change 0.7->06
    my_pal = {"0.7": "#778da9", "0.8": "#415a77", "1.0":"#1b263b"}

    # Get the relevant dataframe
    df_filtered = filter_dataframe(df_by_lvl, level=lvl, fractions=fractions)
    df_filtered['fraction'] = df_filtered['fraction'].astype(str)
    df_filtered['class'] = df_filtered['class'].astype(str)
    #df_long = df_filtered[].melt(id_vars=['', 'fraction'])

    frac = 0.8
    df_filtered_frac = df_filtered[df_filtered['fraction'] == str(frac)]

    feature_names = [feature for feature in sorted(df_filtered.columns, key=str.lower) if feature not in irrelevant_features]
    feature_names = feature_names[:] ##### !!!!!

    # Calculate the number of rows and columns for subplots
    num_features = len(feature_names)
    if lvl in ['network', 'node_higher', 'node_lower']:
        num_plots_per_feature = 4  # boxplot, joyplot, histogram, violin plot
    else:
        num_plots_per_feature = 6 # boxplot, joyplot, histogram, violin plot, boxplot, joyplot

    # Create subplots
    fig = plt.figure(constrained_layout=True, figsize=(18, 4 * num_features))
    fig.suptitle(f'Features distribution - {lvl} level')

    subfigs = fig.subfigures(nrows=num_features, ncols=1)

    for i, subfig in enumerate(subfigs):
        feature = feature_names[i]

        subfig.suptitle(feature)
        
        ax = subfig.subplots(nrows=1, ncols=num_plots_per_feature)

        # Plot the boxplot
        sns.boxplot(df_filtered, x=feature, y='fraction', palette=my_pal, ax=ax[0])
        ax[0].set_title('Boxplot')

        # Plot the violin plot
        sns.violinplot(df_filtered, x=feature, y='fraction', palette=my_pal, ax=ax[1])
        ax[1].set_title('Violin Plot')

        # Plot the KDE
        sns.kdeplot(df_filtered, x=feature, hue='fraction', common_norm=False, fill=True, ax=ax[2])
        ax[2].set_title('KDE')

        # Plot the histogram
        sns.histplot(df_filtered, x=feature, hue='fraction', common_norm=False, ax=ax[3])
        ax[3].set_title('Histogram')

        if lvl not in ['network', 'node_higher', 'node_lower']:

            # Plot the boxplot
            sns.boxplot(df_filtered_frac, x=feature, y='class', ax=ax[4])
            ax[4].set_title(f'Boxplot of {int(frac*100)}% \nobserved network')

            # Plot the KDE
            sns.kdeplot(df_filtered_frac, x=feature, hue='class', fill=True, common_norm=False, ax=ax[5])
            ax[5].set_title(f'KDE of {int(frac*100)}% \nobserved network')

    # Show the plot
    plt.show()


####################################################################################

def plot_miss_val(df, string):
    
    plt.style.use('fivethirtyeight')
    isna = df[df.columns[df.isnull().any()]]
    missing = isna.isnull().sum().sort_values(ascending=False)
        
    fig,ax = plt.subplots(figsize=(10, 7) )
    #plt.style.use('fivethirtyeight')
    ax.set(ylabel = 'Percentage of missing values', xlabel= 'Features',
          title= 'Percentage of missing values of '+string, )
    plt.xticks(rotation=45, ha='right')
    sns.barplot(x=missing.keys(), y=(missing.values/len(df))*100)
    sns.set(style='white', font_scale=1)
    plt.tight_layout()
    plt.show()
    plt.style.use('default')

####################################################################################


def cast2binary(df, FN = 1):
    
    df = pd.DataFrame(df.copy())
    df["class"][df["class"] == "TP"] = 1
    df["class"][df["class"] == "FN"] = FN
    df["class"][df["class"] == "TN"] = 0
    df["class"] = pd.to_numeric(df["class"])
    #df["class"]=df.astype('int')
    
    return df

def corr(df, ignore = [], between='all', plot=True):
    
    df = df.copy()
    
    if ignore:
        df.drop(ignore, inplace=True, axis=1, errors='ignore')
        
    if between=='all':
        
        df.pop('class')
        corr_matrix = df.corr()
        
        if plot == True:
            fig, ax = plt.subplots(figsize=(13,13))
            cmap = sns.diverging_palette(220, 10, as_cmap=True, sep=100)
            sns.heatmap(corr_matrix , cmap = cmap
                        , vmin=-1, vmax=1, center=0)
            plt.rc('xtick', labelsize=10) 
            plt.rc('ytick', labelsize=10)
            plt.title('Correlation matrix of features',  fontsize=20)
            plt.tight_layout()
            
            plt.show()
            plt.close()
        
        corr_matrix = corr_matrix.mask(np.tril(np.ones(corr_matrix.shape)).astype(np.bool)) # Take only upper triangle
        corr_matrix = corr_matrix.unstack().sort_values(ascending=False).reset_index().dropna() # Sort top values
        corr_matrix.columns = ['feature_1','feature_2','value']
        
    elif between=='target':
        
        df = cast2binary(df, FN = 1)
        corr_matrix = df[df.columns[df.columns != 'class']].corrwith(df["class"]).to_frame()
        corr_matrix = corr_matrix.dropna().reset_index() # Sort top values
        corr_matrix.columns = ['feature','value']
        corr_matrix = corr_matrix.sort_values(ascending=False, by='value')
        
        if plot == True:
            
            plt.figure(figsize=(5,15))
            sns.barplot(y=corr_matrix.feature, x=corr_matrix.value, orient='h')
            plt.xlim([-1,1])
            plt.xlabel("Correlation with target variable")
            plt.title('Correlation matrix of features',  fontsize=20)
            plt.tight_layout()
            plt.show()
    
    return corr_matrix

####################################################################################

def target_dist(df):

    # Target distibution - pie chart
    colors = {'TN':'#4c72b0', 'TP':'#55a868', 'FN':'#c44e52'}
    
    df_grouped = df.groupby(['class'])['class'].count()
    df_grouped.plot(kind='pie', figsize=(8, 8), fontsize=15, title = 'Target variable', wedgeprops={'alpha':0.8, 'edgecolor':'k'},
                    textprops={'fontweight':'bold'}, colors = [colors[class_] for class_ in df_grouped.index], autopct="%.1f%%", 
                    pctdistance=0.8, labeldistance=1.1, radius=1).legend(title = "Classes:")
    plt.style.use('fivethirtyeight')
    plt.show()
    
    # Target distibution - histogram
    sns.set(style='whitegrid', palette="deep", font_scale=1.1, rc={"figure.figsize": [8, 5]})
    sns.histplot(df, x='class', hue='class', palette=colors, edgecolor='k').set(title = 'Target variable distibution', ylabel = '#samples')
    plt.style.use('fivethirtyeight')
    plt.show()
    
    plt.style.use('default')

####################################################################################

#def pca

####################################################################################

def plot_pca(df_pca, hue_by, centrs=None, n_dim=2):
    fig = plt.figure(figsize=(14,12))
    fig_projection = '3d' if n_dim > 2 else 'rectilinear'
    ax = fig.add_subplot(111, projection=fig_projection)
    fig.patch.set_facecolor('white')
    label_list = np.unique(hue_by)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if n_dim > 2:
        for l in range(len(label_list)):
            label = label_list[l]
            ix=np.where(hue_by==label)
            ax.scatter(df_pca[ix,0], df_pca[ix,1], df_pca[ix,2], label=str(label), s = 5, alpha=0.1)
            if centrs != None:
                ax.scatter(centrs[l,0], centrs[l,1], centrs[l,2],c="k",marker="X", s=500, label= "K-means clusters centers" if label == label_list[-1] else "")
        ax.view_init(20, 280)
        ax.set_zlabel('PC3') 
        
    else:
        sns.scatterplot( x= df_pca[:,0], y= df_pca[:, 1], hue=hue_by, legend='full',palette = colors[:len(label_list)])
        if centrs != None:
            plt.scatter(x= centrs[:,0],y = centrs[:,1], s=300, c="k",marker="X", label= "K-means clusters centers" )
    ax.set( title= "PCA - by Actual Labels", ylabel='PC2', xlabel='PC1')
    
    lgnd = ax.legend(loc="upper right", scatterpoints=1, fontsize=13)
    for handle in lgnd.legendHandles:
        handle.set_sizes([40.0])
        handle.set_alpha(1)
        
    ax.grid()
    plt.plot()
    
def top_pca_features(pca, df, n=6):
    print(f'Top {n} most important features in each component')
    print('===============================================')
    pca_components = abs(pca.components_)
    for row in range(pca_components.shape[0]):
        # get the indices of the top 6 values in each row
        temp = np.argpartition(-(pca_components[row]), n)

        # sort the indices in descending order
        indices = temp[np.argsort((-pca_components[row])[temp])][:n]

        # print the top 6 feature names
        print(f'Component {row+1}: {df.columns[indices].to_list()}')

def scree_plot(pca, components):
    fig = plt.figure(figsize=(5,4))
    plt.plot(range(1,components+1), pca.explained_variance_ratio_, 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Proportion of Variance Explained')
    plt.xticks(np.arange(1,components+1,1))
    plt.autoscale()
    plt.show()
    
    fig = plt.figure(figsize=(5,4))
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree.cumsum())
    #plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("Principal Components")
    plt.ylabel("Explained variance (%)")
    plt.title("Scree Plot")
    plt.show()
    
####################################################################################
