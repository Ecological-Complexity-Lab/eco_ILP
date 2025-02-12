import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#from debugger_functions import setGlobal, stop
# from helper.model_evaluation import evaluate, evaluate_model, grouped_evaluation, overall_evaluation

#from sklearn.metrics import make_scorer ###!!!
from sklearn.model_selection import GroupShuffleSplit

# tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow's debugging logs https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information/42121886#42121886


# Pipeline
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder

# models
import xgboost as xgb
import lightgbm as lgb
# from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, GroupKFold, PredefinedSplit
input_dim = 1 # temp | DNN


####################################################################################

def features_cast(df, config):
    
    cast_dict = {}
    
    for feature in df.columns:
        
        type_old = str(df[feature].dtype)
        feature_props = config['features'][feature]
        type_new = feature_props[config['features_props']['type']]
        
        if type_new == 'bool':
            type_new = 'int64'
        
        if type_old != type_new:
            cast_dict[feature] = type_new
    
    return df.astype(cast_dict) # Ignore nan conversion

def preprocess(df, target='class'):
    
    # Drop unrelated columns
    df = df.drop(['weight', 'lower_level', 'higher_level'], axis=1)

    # Deal with NaN # TODO: verify for each feature what is the best approach
    df = df.fillna(0)

    # Cast type of features # TODO: load from features metadata file
    # df = features_cast(df, config)

    # Move target to last position
    df = df[ [ col for col in df.columns if col != target ] + [target] ]
    
    return df
####################################################################################

from sklearn.preprocessing import OrdinalEncoder

def cast2numeric(X_train, X_test): 
    
    trans_cols = list(X_train.select_dtypes(['object']).columns)
    
    enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=100)
    enc.fit(X_train[trans_cols])
    X_train[trans_cols] = enc.transform(X_train[trans_cols])
    X_test[trans_cols] = enc.transform(X_test[trans_cols])
        
    return X_train, X_test

####################################################################################

def fill_missing_vals(df):
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing.loc[(missing!=0)]
    for feature in missing.index:
        if (df[feature].dtypes == np.float64 or df[feature].dtypes == np.int64):
            df[feature].fillna(0, inplace=True)
        else:
            df[feature].fillna("0", inplace=True)
    return df


####################################################################################

from sklearn.metrics import f1_score

def custom_scorer(y_true, y_proba, y_class):
    
    y_merged = pd.merge(y_true, y_class, how="left", left_index=True, right_index=True)
    y_merged = pd.concat([y_merged.reset_index(drop=True), pd.DataFrame(y_proba)], axis=1)
    y_merged.rename(columns={y_merged.columns[0]: 'y_true', y_merged.columns[2]: 'y_proba'}, inplace=True)
    
    y_filtered = y_merged[(y_merged["class"] == "TN") | (y_merged["class"] == "FN")]
    
    #return roc_auc_score(y_filtered["y_true"], y_filtered["y_proba"])
    return f1_score(y_filtered["y_true"], y_filtered["y_proba"])

    
####################################################################################

from matplotlib import rcParams
  
def plot_history(history):
    
    rcParams['figure.figsize'] = (18, 8)
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
        
    fig, ax = plt.subplots()
    
    plt.plot(history.epoch, history.history['loss'], label='Loss')
    plt.plot(history.epoch, history.history['accuracy'], label='Accuracy')
    plt.plot(history.epoch, history.history['precision'], label='Precision')
    plt.plot(history.epoch, history.history['recall'], label='Recall')
    
    plt.title('Evaluation metrics', size=20)
    plt.xlabel('Epoch', size=14)
    plt.legend()
    plt.show()

####################################################################################

def plot_metrics(history):
    
    rcParams['figure.figsize'] = (12, 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    metrics = ['accuracy', 'loss', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
    
        plt.legend()
    plt.show()
    
####################################################################################

def plot_search_results(search):
    
    cv_df = pd.DataFrame(search.cv_results_)
    results = ['mean_test_score',
               'mean_train_score',
               'std_test_score', 
               'std_train_score']
    params=search.param_grid

    def pooled_var(stds):
        # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
        n = 5 # size of each group
        return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))

    fig, axes = plt.subplots(1, len(params), 
                             figsize = (5*len(params), 7),
                             sharey='row')
    axes[0].set_ylabel("Score", fontsize=25)


    for idx, (param_name, param_range) in enumerate(params.items()):
        grouped_df = cv_df.groupby(f'param_{param_name}')[results]\
            .agg({'mean_train_score': 'mean',
                  'mean_test_score': 'mean',
                  'std_train_score': pooled_var,
                  'std_test_score': pooled_var})

        previous_group = cv_df.groupby(f'param_{param_name}')[results]
        axes[idx].set_xlabel(param_name, fontsize=15)
        axes[idx].set_ylim(0.0, 1.1)
        lw = 2
        
        
        axes[idx].plot(param_range, grouped_df['mean_train_score'], label="Training score",
                    color="darkorange", lw=lw)
        axes[idx].fill_between(param_range,grouped_df['mean_train_score'] - grouped_df['std_train_score'],
                        grouped_df['mean_train_score'] + grouped_df['std_train_score'], alpha=0.2,
                        color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df['mean_test_score'], label="Cross-validation score",
                    color="navy", lw=lw)
        axes[idx].fill_between(param_range, grouped_df['mean_test_score'] - grouped_df['std_test_score'],
                        grouped_df['mean_test_score'] + grouped_df['std_test_score'], alpha=0.2,
                        color="navy", lw=lw)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=40)
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=20)
    fig.subplots_adjust(bottom=0.25, top=0.85)  
    plt.show()

def plot_search_results2(grid):
    """
    Params: 
        grid: A trained GridSearchCV object.
    """
    ## Results from grid search
    results = grid.cv_results_
    means_test = results['mean_test_score']
    stds_test = results['std_test_score']
    means_train = results['mean_train_score']
    stds_train = results['std_train_score']

    ## Getting indexes of values per hyper-parameter
    masks=[]
    masks_names= list(grid.best_params_.keys())
    for p_k, p_v in grid.best_params_.items():
        masks.append(list(results['param_'+p_k].data==p_v))

    params=grid.param_grid
    
    ## Ploting results
    fig, ax = plt.subplots(1,len(params),sharex='none', sharey='all',figsize=(20,5))
    fig.suptitle('Validation curves')
    fig.text(0.04, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    pram_preformace_in_best = {}
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i+1:])
        pram_preformace_in_best
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(params[p])
        y_1 = np.array(means_test[best_index])
        e_1 = np.array(stds_test[best_index])
        y_2 = np.array(means_train[best_index])
        e_2 = np.array(stds_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test')
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^',label='train' )
        ax[i].set_xlabel(p.upper())

    plt.legend()
    plt.show()
    
def plot_learning_curve(train_sizes, train_scores, test_scores, title):
    fig, ax = plt.subplots()
    ax.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Accuracy')
    ax.plot(train_sizes, np.mean(test_scores, axis=1), linestyle='--', label='Validation Accuracy')
    ax.set( ylim= [0,1], title = "Learning Curve", xlabel ='Training Data Size' ,  ylabel = 'Model accuracy' )
    ax.grid()
    ax.legend(loc='lower right')
    plt.show()


def plot_learning_curve2(train_sizes, train_scores, test_scores, title, alpha=0.1):
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    plt.plot(train_sizes, train_mean, label='train score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean + train_std,
                     train_mean - train_std, color='blue', alpha=alpha)
    plt.plot(train_sizes, test_mean, label='test score', color='red', marker='o')

    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, color='red', alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of training points')
    plt.ylabel('Accuracy')
    plt.grid(ls='--')
    plt.legend(loc='best')
    plt.show()
        
        
####################################################################################

        
import shap
import warnings

def feature_importances(pipelines, X_train_transformed=None, plot=True):
    
    feat_importance_full = pd.DataFrame()
    
    for model_name, pipe in pipelines.items():
        
        
        if pipe.__class__.__name__.endswith('SearchCV'): # If used RandomizedSearchCV or GridSearchCV
            
            # Get fitted model
            fitted_model = pipe.best_estimator_._final_estimator
            
            # Best Score
            # search.best_score_
            
            # Best Parameters
            best_params = pipe.best_params_
            
            # Get feature names
            feature_names = pipe.best_estimator_['preprocessor'].get_feature_names_out()
        else:
            
            # Get fitted model
            fitted_model = pipe['classifier']
        
            # Best Parameters
            # best_params(estimator)
            
            # Get feature names
            feature_names = pipe['preprocessor'].get_feature_names_out()


        if model_name != 'KerasClassifier':

            # Feature Importance
            fitted_model.feature_names = feature_names
            feat_importance = top_features(fitted_model, n=25, print_ = False, plot = plot)

            # Best Parameters
            #best_params(estimator)

            feat_importance_full = pd.concat([feat_importance_full, feat_importance], axis=1).rename(columns={0:model_name})

            # Shap

            # X_train_sample = X_train_transformed#.sample(1000, random_state=10)

            # if model_name == 'LogisticRegression':
            #     explainer = shap.LinearExplainer(fitted_model, X_train_sample, feature_perturbation="interventional")
            #     shap_values = explainer.shap_values(test_shap)

        else:
            def f_wrapper(X):
                return fitted_model.predict(X).flatten()

            X_train_sample = X_train_transformed.sample(100, random_state=10)
            X_train_summary = shap.kmeans(X_train_sample, 20)
            explainer = shap.KernelExplainer(f_wrapper,X_train_summary)
            with warnings.catch_warnings(): ## deal with warning!!
                warnings.filterwarnings("ignore")
                shap_values  = explainer.shap_values(X_train_sample, verbose=0)
            shap.summary_plot(shap_values, X_train_sample, plot_type="bar", max_display=20)

            mean_shap_feature_values_sorted = pd.DataFrame(shap_values, columns=X_train_transformed.columns).abs().mean(axis=0).sort_values(ascending=False)
            mean_shap_feature_values = pd.DataFrame(shap_values, columns=X_train_transformed.columns).abs().mean(axis=0)
            feat_importance_full = feat_importance_full.merge(mean_shap_feature_values.rename(model_name), left_index=True, right_index=True)


            #mean_shap_feature_values.to_csv('features_DNN.csv')
                    
    return feat_importance_full.sort_values(by=feat_importance_full.columns[-1])

def feature_importances2(ml_list, X_train_transformed=None, plot=True):
    
    feat_importance_full = pd.DataFrame()
    
    for model_name, pipe in ml_list:
        
        model_name = ml.model_name
        # pipe=

        if pipe.__class__.__name__.endswith('SearchCV'): # If used RandomizedSearchCV or GridSearchCV
            
            # Get fitted model
            fitted_model = pipe.best_estimator_._final_estimator
            
            # Best Score
            # search.best_score_
            
            # Best Parameters
            best_params = pipe.best_params_
            
            # Get feature names
            feature_names = pipe.best_estimator_['preprocessor'].get_feature_names_out()
        else:
            
            # Get fitted model
            fitted_model = pipe['classifier']
        
            # Best Parameters
            # best_params(estimator)
            
            # Get feature names
            feature_names = pipe['preprocessor'].get_feature_names_out()


        if model_name != 'KerasClassifier':

            # Feature Importance
            fitted_model.feature_names = feature_names
            feat_importance = top_features(fitted_model, n=25, print_ = False, plot = plot)

            # Best Parameters
            #best_params(estimator)

            feat_importance_full = pd.concat([feat_importance_full, feat_importance], axis=1).rename(columns={0:model_name})

            # Shap

            # X_train_sample = X_train_transformed#.sample(1000, random_state=10)

            # if model_name == 'LogisticRegression':
            #     explainer = shap.LinearExplainer(fitted_model, X_train_sample, feature_perturbation="interventional")
            #     shap_values = explainer.shap_values(test_shap)

        else:
            def f_wrapper(X):
                return fitted_model.predict(X).flatten()

            X_train_sample = X_train_transformed.sample(100, random_state=10)
            X_train_summary = shap.kmeans(X_train_sample, 20)
            explainer = shap.KernelExplainer(f_wrapper,X_train_summary)
            with warnings.catch_warnings(): ## deal with warning!!
                warnings.filterwarnings("ignore")
                shap_values  = explainer.shap_values(X_train_sample, verbose=0)
            shap.summary_plot(shap_values, X_train_sample, plot_type="bar", max_display=20)

            mean_shap_feature_values_sorted = pd.DataFrame(shap_values, columns=X_train_transformed.columns).abs().mean(axis=0).sort_values(ascending=False)
            mean_shap_feature_values = pd.DataFrame(shap_values, columns=X_train_transformed.columns).abs().mean(axis=0)
            feat_importance_full = feat_importance_full.merge(mean_shap_feature_values.rename(model_name), left_index=True, right_index=True)


            #mean_shap_feature_values.to_csv('features_DNN.csv')
                    
    return feat_importance_full.sort_values(by=feat_importance_full.columns[-1])