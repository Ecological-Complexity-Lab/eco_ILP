import os
import yaml
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score#, top_k_accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, log_loss, brier_score_loss, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, make_scorer
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.inspection import permutation_importance

# Pipeline
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.utils import shuffle

# models
import xgboost as xgb
# import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest#, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.neural_network import MLPClassifier
# from sklearn import svm
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier


# cross-validation
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, GroupShuffleSplit, GroupKFold, StratifiedGroupKFold

# -----------------------------
# DNN
# input_dim = 1
# import kerastuner as kt
# import tensorflow as tf
# from kerastuner.tuners import BayesianOptimization
# from keras.wrappers.scikit_learn import KerasClassifier
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow's debugging logs
# from helper.DNN import create_model, create_model_cv, dnn_tuner

# -----------------------------
# import shap
import warnings

import warnings # TODO: check if this code can be here or in the main file
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning) # turn off some annoying warnings (LogisticRegression's convergence warning)

from helper.scorer import custom_scorer

class LinkPredict():

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
    classifiers = {
        ''
            'LogisticRegression':LogisticRegression(max_iter=300, n_jobs=-1), # Logistic Regression
            'RandomForestClassifier':RandomForestClassifier(n_jobs=-1), # Random Forest 
            # 'KNeighborsClassifier':KNeighborsClassifier(n_jobs=-1), # KNN
            'XGBClassifier':xgb.XGBClassifier(n_jobs=-1), # XGBoost
            # 'KerasClassifier':KerasClassifier(build_fn=create_model, epochs=5), # DNN | dummy model
            # 'LGBMClassifier':lgb.LGBMClassifier(n_jobs=-1), # LightGBM
            # 'BayesianOptimization':BayesianOptimization(create_model_cv, # DNN cv (testing)
            #                                             objective=kt.Objective('val_balanced_accuracy', direction="max"),
            #                                             max_trials=10, #10
            #                                             executions_per_trial=3, #3?
            #                                             overwrite=True,
            #                                             # directory='my_dir',
            #                                             # project_name='my_project'
            #                                             ),
            # 'DecisionTreeClassifier':DecisionTreeClassifier(),
        }
    
    def __init__(self, classifier_name='RandomForestClassifier', estimators=None, params=None):
        self.train_link_id = None
        self.test_link_id = None
        self.model = self.set_classifier(classifier_name, estimators)
        self.model_name = self.model.__class__.__name__
        self.params_dist = self.set_default_params(classifier_name)
        self.trained_model = None
        self.preprocessor = None
        self.class_weight = None
        self.optimaizer = None

        self.trained_models = []
        self.test_link_ids = []

        if params is not None:
            self.set_params(params)

    def set_classifier(self, classifier_name='RandomForestClassifier', estimators=None, train_size=None):
        if classifier_name=='VotingClassifier':
            return VotingClassifier(estimators)
        return self.classifiers[classifier_name]
    
    def set_default_params(self, classifier_name='RandomForestClassifier'):
        # default params of each classifier, return the params of the given classifier
        
        ml_params = {
            'LogisticRegression':
                [
                    {'penalty': ['l2', 'none'],
                    'solver': ['newton-cg', 'lbfgs', 'sag'],
                    'C': [100, 10, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001],
                    'max_iter': [300]}, # [30, 60, 100, 200]
                    {'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'C': [100, 10, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001],
                    'max_iter': [300]}, # [30, 60, 100, 200]
                    {'penalty': ['elasticnet', 'l1', 'l2'],
                    'solver': ['saga'],
                    'C': [100, 10, 1.0, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001],
                    'max_iter': [300]} # [30, 60, 100, 200]
                ],

            'RandomForestClassifier':
                {'n_estimators' : [10, 15, 20, 50, 100, 300],
                'min_samples_split': [1 ,2, 3, 4, 5, 10],
                'min_samples_leaf': [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
                'max_samples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'max_leaf_nodes': [2, 4, 8, 16, 32, 64, 128],
                'max_features' : ['sqrt', 'log2'],
                'max_depth': [3, 5, 7, 10, 20, 30, 40, 50, 60, None],
                'criterion' : ["gini", "entropy"],
                'bootstrap': [True], #False
                },

            'XGBClassifier':
                {'learning_rate': [0.001, .01, .05, 0.1, 0.2, 0.3],
                 'tree_method': ['hist'], # 'exact', 'approx', 'hist', 'gpu_hist'
                 'n_estimators': np.arange(10, 110, 10).tolist(),
                 'max_depth': np.arange(1, 20, 2).tolist()+[None],
                 'objective': ['binary:logistic'],
                 'subsample': np.arange(0.1, 1.1, 0.1).tolist(),
                 'gamma': [0, 0.1, 0.3, 0.5, 1],
                 'reg_alpha': [0, 1e-5, 1e-2, 0.1, 1, 100],
                 'reg_lambda': [0, 1e-5, 1e-2, 0.1, 1, 100],
                 'booster': ['gbtree'],
                 'colsample_bytree': np.arange(0.1, 1.1, 0.1).tolist(),
                # 'min_child_weight' : [1, 5, 7],
                # 'scale_pos_weight': [1],
                },
            
            'DecisionTreeClassifier':None,
            'VotingClassifier':None,
            'SVC':
                {'kernel' : ['linear', 'rbf', 'poly', 'sigmoid'],
                'C': [0.1, 1, 10, 100, 1000],
                'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                #'degree' : [0, 1, 2, 3, 4, 5, 6], #use only with poly
                
                },
            'KNeighborsClassifier':
                {'leaf_size' : list(range(1,50)),
                'n_neighbors': list(range(1,30)),
                'p': [1,2],
                'weights': ['uniform', 'distance'],
                'metric' : ['minkowski','euclidean','manhattan']             
                },
            # 'KerasClassifier':{},
            # 'BayesianOptimization':{},
            }
        
        return ml_params[classifier_name]
    
    def set_transformer(self, df, columns_to_ignore=[], save=True, return_preprocessor=False):

        # Save feature names into types
        numeric_features = list(df.select_dtypes(exclude=['object']).columns)
        categorical_features = list(df.select_dtypes(include=['object']).columns)

        # Drop some column - they were needed only for previous steps but should not be features
        numeric_features = [col for col in numeric_features if col not in columns_to_ignore]
        categorical_features = [col for col in categorical_features if col not in columns_to_ignore]

        # Set transformers
        numeric_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy="mean")), 
                                                ("scaler", QuantileTransformer())]) #MinMaxScaler #StandardScaler #RobustScaler #QuantileTransformer
        
        known_categories = []

        for feature in categorical_features:
            known_categories.append(list(df[feature].unique()))

        categorical_transformer = OneHotEncoder(categories=known_categories, 
                                                handle_unknown="infrequent_if_exist", 
                                                drop='if_binary')

        # Set the whole preprocessor step
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categoric", categorical_transformer, categorical_features),
            ], verbose_feature_names_out=False, remainder='drop'
        )

        # Save preprocessor
        if save:
            self.preprocessor = preprocessor
        if return_preprocessor:
            return preprocessor

    def fit(self, X_train=None, y_train=None, df=None, dataset_link_id=None, cv=None, cv_inner=None, cv_outer=None, class_weight = 'balanced', select=None, tuner_name=None, scorer='f1', columns_to_ignore=[]):

        ## Initial checks

        # Verify X_train and y_train
        if X_train is None or y_train is None:
            X_train, y_train = self.subset_data(df, dataset_link_id)
        
        # Verify correct cv or nested cv usage           
        if (cv_inner is not None) or (cv_outer is not None):
            if (cv_inner is None) or (cv_outer is None):
                raise Exception('Please provide both cv_inner and cv_outer, or provide "cv" argument in the case of no nested cv')
            nested_cv = True
            cv = cv_inner
        else:
            nested_cv = False

        # Get the number of splits in cv, if any
        if cv:
            cv_n_splits = cv.get_n_splits()
        else:
            cv_n_splits = 0

        # Verify tuner
        if cv_n_splits > 0:
            if tuner_name == None:
                print('No tuner name provided. RandomizedSearchCV was chosen by default.')
                tuner_name = 'RandomizedSearchCV'
        else:
            if tuner_name != None:
                print('Tuner name provided but cv is set to 0. Tuner will not be used.')
                tuner_name = None

        # Feature selection
        if select is not None:
            unselect = [s for s in X_train.columns if (s not in select)]
            X_train = X_train.drop(unselect, axis=1)

        # Create preprocessor
        self.set_transformer(X_train, columns_to_ignore=columns_to_ignore)

        # Avoid errors during fitting, happens when using cv in some cases
        X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)

        # Deal with existing-link instances | It might be better to deal with it here in the custom scorer rather in in the costume cv, as passing y_test with 3 classes (1,0,-1) to a classifier will cause him to initiate a 3-class classifier
        if cv_n_splits > 0:
            cv.trueLinks_id = X_train[y_train == 1]['link_ID'] # Keep original existing links instances for later
            if cv_outer:
                cv_outer.trueLinks_id = X_train[y_train == 1]['link_ID']
        y_train[y_train == -1] = 1 # Relabel subsampled links as existing-links

        # New - needs testing with multiple classifiers
        self.model.class_weight = 'balanced'

        # Create pipeline
        pipe = Pipeline(steps=[("preprocessor", self.preprocessor), 
                               ("classifier", self.model)])
        
        # Choose a scorer
        # scorer = 'f1'
        scorer = make_scorer(custom_scorer, greater_is_better=True, needs_proba=False)
        
        # Set the parameters of the model and CV tuner
        if isinstance(self.params_dist, dict):
            params_dist = {f'classifier__{k}': v for k, v in self.params_dist.items()} # Rename each key in the dictionary to 'classifier__' + key
        elif isinstance(self.params_dist, list):
            params_dist = {f'classifier__{k}': v for params_dict in self.params_dist for k, v in params_dict.items() }

        tuner_params = {
            # 'param_distributions':params_dist,
            'scoring':scorer,
            'cv' : cv, #GroupKFold(n_splits=cv),
            'return_train_score':True,
            'n_jobs':-1,
            'verbose':1,
        }

        groups = X_train['name'] # groups of train data | TODO: 'name' is not general, the info is in cv.group_by but not always
        
        fit_params = {
            'groups':groups,
            # 'classifier__sample_weight' : compute_sample_weight(class_weight = class_weight, y = y_train), # wrong place to calc this, as later there are processes such as undersampling which will change the proportions of the classes
        }

        if cv_n_splits == 0:

            # No need for groups in case of no cv
            fit_params.pop('groups')

            # Fit the model 
            pipe.fit(X_train, y_train, **fit_params) #, classifier__callbacks = [history] (dnn)

            # Save the trained model
            self.trained_model = pipe

        else:

            # In case of not DNN
            if self.model_name not in ['RandomSearch', 'BayesianOptimization']:

                # Fit the model using the chosen tuner
                if tuner_name == 'RandomizedSearchCV':
                    tuner = RandomizedSearchCV(pipe, param_distributions=params_dist, **tuner_params)
                elif tuner_name == 'GridSearchCV':
                    tuner = GridSearchCV(pipe, param_grid=params_dist, **tuner_params)
                
                if nested_cv:
                    
                    for train_index, test_index in cv_outer.split(X_train, y_train):
                        
                        X_train_inner, y_train_inner = X_train.loc[train_index], y_train.loc[train_index]

                        tuner = RandomizedSearchCV(pipe, param_distributions=params_dist, **tuner_params)
                        
                        tuner.fit(X_train_inner.reset_index(drop=True), y_train_inner.reset_index(drop=True))
                        
                        self.trained_models.append(tuner)
                        
                else:
                    # Fit the model
                    tuner.fit(X_train, y_train) #, **fit_params

            else:
                pass

            # Save the tuner
            if not nested_cv:
                self.trained_model = tuner
        
        # Save the preprocessor
        self.preprocessor = pipe['preprocessor']
    
    def available_classifiers(self):
        classifiers_names = list(self.classifiers.keys())
        return classifiers_names

    def subset_data(self, df, dataset_link_id=None, cast_target=True, return_true_classes=False):
        
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
    
    def set_fold_model(self, fold):
        self.trained_model = self.trained_models[fold]

    def get_transformed_data(self, X=None, df=None, dataset_link_id=None, keep_features=True, feaure_names=True, fit=False):

        if X is None:
            X, _ = self.subset_data(df, dataset_link_id)

        # Transform X using the preprocessor
        if fit:
            X_transformed = self.preprocessor.fit_transform(X)
        else:
            X_transformed = self.get_fitted_preprocessor().transform(X)
        
        # Get the set of feature names from the preprocessor
        preprocessor_feature_names = self.get_fitted_preprocessor().get_feature_names_out()

        # Create a new DataFrame with the transformed columns and feature names
        if feaure_names:
            try:
                X_transformed = pd.DataFrame(X_transformed.toarray(), columns=preprocessor_feature_names) 
            except AttributeError:
                X_transformed = pd.DataFrame(X_transformed, columns=preprocessor_feature_names)

        if not keep_features:

            # Filter the columns of X_transformed to keep only those in preprocessor_feature_names
            columns_to_keep = [col for col in X_transformed.columns if col in preprocessor_feature_names]

            # Reorder the columns to match the original order in preprocessor_feature_names
            X_transformed = X_transformed[columns_to_keep]
        
        return X_transformed
        
    def get_params(self):
        return self.params_dist
    
    def set_params(self, params):
        self.params = params
        self.model.set_params(**params)

    def get_trained_model(self):

        if self.trained_model.__class__.__name__.endswith('CV'):
            return self.trained_model.best_estimator_._final_estimator

        return self.trained_model

    def get_fitted_preprocessor(self):

        # Get trained model
        trained_model = self.trained_model 

        # todo: if trained_model=none, raise error

        if trained_model.__class__.__name__.endswith('CV'):

            # Get the fitted pipeline
            fitted_pipeline = trained_model.best_estimator_

            fitted_preprocessor = fitted_pipeline.named_steps['preprocessor'] # Get the fitted preprocessor
        else:
            fitted_preprocessor = self.preprocessor
        
        return fitted_preprocessor

    def feature_importance(self, n=None, X_train=None):
                
        trained_model = self.trained_model # Get trained model

        if trained_model.__class__.__name__.endswith('CV'):
            trained_model = trained_model.best_estimator_._final_estimator # check
        elif self.trained_model.__class__.__name__ == 'Pipeline':
            trained_model = trained_model._final_estimator

        fitted_preprocessor = self.get_fitted_preprocessor() # Get fitted preprocessor

        # Get feature names
        feature_names = fitted_preprocessor.get_feature_names_out() 

        if n is None:
            n=feature_names.shape[0]
        
        if self.model_name == 'LogisticRegression':
            importance  = trained_model.coef_[0]
            feature_importance = pd.Series(importance, index=feature_names)
            feature_importance = feature_importance[feature_importance.abs().nlargest(n).index].sort_values()
        
        elif self.model_name == 'DecisionTreeClassifier':
            pass
        # elif self.model_name in ['KerasClassifier', 'BayesianOptimization']:
        #     pass
        else:
            importance  = trained_model.feature_importances_
            feature_importance = pd.Series(importance, index=feature_names).nlargest(n)

        return feature_importance
    
    def permutation_importance(self, X, y, scoring='f1', n_repeats=3):

        pfi_result = permutation_importance(self.trained_model, 
                                            X, 
                                            y,
                                            scoring=scoring, 
                                            n_repeats=n_repeats,
                                            random_state=0, 
                                            n_jobs=-1)
        return pfi_result

    def plot_feature_importance(self, n=None, labels_inside = False, ax=None, show = True):
        
        feature_importance = self.feature_importance(n)

        if n is None:
            n=self.get_fitted_preprocessor().get_feature_names_out().shape[0]

        # Get trained model
        trained_model = self.trained_model 
        if trained_model.__class__.__name__ == 'Pipeline':
            trained_model = trained_model._final_estimator

        # Set y_label
        y_label = 'Importance'
        if self.model_name == 'LogisticRegression':
            y_label = 'Coef'
        
        # Create figure
        if ax is None: # if ax is not a part of a subplot
            fig, ax = plt.subplots(figsize=(8, n/3))
        else:
            labels_inside = True

        y_pos = np.arange(len(feature_importance))

        # Create a horizontal bar plot
        bars = ax.barh(y_pos, feature_importance, align='center')

        # Set the y-axis ticks and labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_importance.index)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel('Feature Name')
        ax.set_ylabel(y_label)
        ax.set_title(f'Feature Importance ({self.model_name} - top {n} features)')

        if labels_inside:
            # Move the labels inside the bars
            for index, bar in enumerate(bars):
                ax.text(0, bar.get_y() + bar.get_height() / 2,
                    feature_importance.index[index], color='black', ha='left', va='center')
            ax.set_yticklabels([])  # Remove y-axis tick labels

        # Plot
        if show:
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return feature_importance
    
    def plot_permutation_importance(self, X=None, y=None, pfi_result=None, scoring='f1', n_repeats=3, ax=None, show=True):
        """Plot Permutation Feature Importance."""
        if pfi_result is None:
            pfi_result = self.permutation_importance(X, y, scoring, n_repeats)

        sorted_idx = pfi_result.importances_mean.argsort()
        features_names = X.columns

        plt.figure(figsize=(10, len(features_names) // 2))
        plt.boxplot(pfi_result.importances[sorted_idx].T, vert=False,
                    labels=np.array(features_names)[sorted_idx])
        plt.title(f"Permutation Feature Importance for {self.model_name}, ({scoring}) score")
        plt.show()
        plt.close()
    
    def predict_proba(self, X=None, df=None, dataset_link_id=None):

        if X is None:
            X, _ = self.subset_data(df, dataset_link_id)

        # Get the name of the model
        model_name = self.model_name

        # Calculate probabilities for the given model
        if model_name in ['LinearSVC', 'PassiveAggressiveClassifier']:
            d = self.trained_model.decision_function(X)
            y_proba = np.exp(d) / np.sum(np.exp(d))
        elif model_name == 'RandomSearch' or model_name == 'BayesianOptimization': # DNN
            pass

        else:
            y_proba = self.trained_model.predict_proba(X)[:,1]

        return y_proba
    
    def available_metrics(self):
        return list(self.metrics_functions.keys())

    def plot_confusion_matrix(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, normalize=True, ax=None, title='Confusion Matrix', show=True, save=False, save_path=None):
        
        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

        # Get predictions based on probabilities, using the given threshold
        y_pred = (y_proba >= threshold).astype('int') 

        # Deal with normalization
        if normalize:
            cm_normalize = 'true'
            title_surffix = '\n(normalized)'
        else:
            cm_normalize = None
            title_surffix = '\n(not normalized)'
    
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize=cm_normalize)

        # Create figure
        if ax is None: # if ax is not a part of a subplot
            fig, ax = plt.subplots(figsize=(8, 8))
        cm_disp = ConfusionMatrixDisplay(cm, display_labels=['non-existing\nlinks (0)','missing\nlinks (1)'])
        cm_disp.plot(ax=ax, cmap = plt.cm.Blues, include_values=True)
        #cm_disp.ax_.grid()
        cm_disp.ax_.set(xlabel='Predicted Class', ylabel='True Class')
        cm_disp.ax_.set_title(title) #+ title_surffix
        
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

    def grouped_evaluation(self, X=None, y_true=None, df=None, dataset_link_id=None, group_by = 'name', threshold=0.5, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss']):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

        # Get predictions based on probabilities, using the given threshold
        y_pred = (y_proba >= threshold).astype('int') 

        # Get groups for test data
        groups_test = np.array(X[group_by].values) 

        # Combine y_pred, y_proba, y_test and groups_test into one dataframe
        y_combined = pd.concat([ 
            pd.DataFrame(y_pred).rename(columns={0:'y_pred'}), 
            pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), 
            pd.DataFrame(y_true).reset_index(drop=True).rename(columns={'class':'y_true'}),
            pd.DataFrame(groups_test).rename(columns={0:'group'})], axis=1)
        
        # Group by group_by
        groups = y_combined.groupby('group') 
        
        # Calculate metrics for each group
        results_by_groups = pd.concat([
            groups.apply(lambda group: self.metrics_functions[metric](group.y_true, group.y_pred, group.y_proba)).rename(metric)
            for metric in metrices
        ], axis=1).reset_index().rename(columns={'index': 'group'}).round({metric: 2 for metric in metrices})
        
        return results_by_groups
    
    def plot_grouped_evaluation(self, X=None, y_true=None, df=None, dataset_link_id=None, group_by = 'name', threshold=0.5, split_by=None, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss'], ax=None, title='Boxplots of metrices per network', show=False):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        results_by_groups = self.grouped_evaluation(X, y_true, group_by=group_by, threshold=threshold, metrices=metrices)

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
    
    def plot_roc_curve(self, X=None, y_true=None, df=None, dataset_link_id=None, split_by=None, threshold=0.5, ax=None, title='ROC Curve', return_curves=False, show=True, save=False, save_path=None):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X)
        
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
            rc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=self.model_name)
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
    
    def plot_pr_curve(self, X=None, y_true=None, df=None, dataset_link_id=None, split_by=None, threshold=0.5, ax=None, title='Precision-Recall Curve', show=True, save=False, save_path=None):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

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
            pr_disp.plot(ax=ax, label=self.model_name+' (area = %0.2f)'% average_precision)
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
    
    def plot_pr_vs_threshold(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, group_by=None, split_by=None, ax=None, title='Precision-Recall vs Threshold', show=True, save=False, save_path=None):
        
        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

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
    
    def plot_probs_distribution(self, X=None, y_true=None, df=None, dataset_link_id=None, axes=None, show=True, save=False, save_path=None):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

        y_new = pd.concat([pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), pd.DataFrame(y_true).reset_index(drop=True)], axis=1)
        
        if axes is None: # if ax is not a part of a subplot
            fig, axes = plt.subplots(1,2, figsize=(10,4))

        ax=axes[0]
        sns.histplot(data=y_new[y_new['class'] == 1], x="y_proba", kde=True, ax=ax).set(title='Missing link probability distribution') 
        ax.axvline(0.5)
        
        ax=axes[1]
        sns.histplot(data=y_new[y_new['class'] == 0], x="y_proba", kde=True, ax=ax).set(title='Non-existing links probability distribution')
        ax.axvline(0.5)
        
        # Plot
        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

        return None
    
    def plot_hypothetic_CI(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, group_by = 'name', title='', show=True, save=False, save_path=None):
        # plot confidence interval using hypothetical missing-links in original data

        # if frac == 'all':
        #     title = f'{self.model_name}, all fractions'
        # else:
        #     title = f'{clf_name}, {100-frac*100:.0f}% removed links'
        title = f'{self.model_name}'

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

        # Get predictions based on probabilities, using the given threshold
        y_pred = (y_proba >= threshold).astype('int') 

        # Get groups for test data
        groups_test = np.array(X[group_by].values) 

        # Combine y_pred, y_proba, y_true and groups_test into one dataframe
        y_combined = pd.concat([ 
            pd.DataFrame(y_pred).rename(columns={0:'y_pred'}), 
            pd.DataFrame(y_proba).rename(columns={0:'y_proba'}), 
            pd.DataFrame(y_true).reset_index().rename(columns={'class':'y_true', 'index': 'sample_index'}),
            pd.DataFrame(groups_test).rename(columns={0:'group'})], axis=1)

        fractions = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    
        error_dict = {
            'ROC AUC':pd.DataFrame(),
            'PR AUC':pd.DataFrame(),
            'f1':pd.DataFrame(),
            'accuracy':pd.DataFrame(),
            'specificity':pd.DataFrame(),
            'precision':pd.DataFrame(),
            'recall':pd.DataFrame(),
        }
        
        
        groups = y_combined.groupby('group')
        
        for f in fractions:
            
            best_scenario_idx = []
            worst_scenario_idx = []
            
            for name, group in groups:
                
                best_scenario = group[(group['y_true']==0) & (group['y_pred']==1)]
                worst_scenario = group[(group['y_true']==0) & (group['y_pred']==0)]
                
                sample_size = int(len(group[group['y_true']==0])*f) # TODO: should be TN + FN !!!!!!!!!!

                best_scenario_max_sample_size = best_scenario.shape[0]
                worst_scenario_max_sample_size = worst_scenario.shape[0]

                best_scenario_idx += list(best_scenario.sample(min(sample_size, best_scenario_max_sample_size), replace=False)['sample_index'])
                worst_scenario_idx += list(worst_scenario.sample(min(sample_size, worst_scenario_max_sample_size), replace=False)['sample_index'])
                
                if sample_size > best_scenario_max_sample_size: # if the amount of missing-links are greater than the actual FNs, then it should harm the best scenerio case
                    best_scenario_idx += list(worst_scenario.sample(sample_size-best_scenario_max_sample_size, replace=False)['sample_index'])
                
            y_true_best_scenario = y_true.copy()
            y_true_worst_scenario = y_true.copy()
            
            y_true_best_scenario['class'][best_scenario_idx] = 1
            y_true_worst_scenario['class'][worst_scenario_idx] = 1

            error_dict['ROC AUC'] = pd.concat([error_dict['ROC AUC'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[roc_auc_score(y_true, y_proba)],
                                                'upper':[roc_auc_score(y_true_best_scenario, y_proba)],
                                                'lower':[roc_auc_score(y_true_worst_scenario, y_proba)],
                                            })], axis=0)
            
            error_dict['PR AUC'] = pd.concat([error_dict['PR AUC'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[average_precision_score(y_true, y_proba)],
                                                'upper':[average_precision_score(y_true_best_scenario, y_proba)],
                                                'lower':[average_precision_score(y_true_worst_scenario, y_proba)],
                                            })], axis=0)
            
            error_dict['f1'] = pd.concat([error_dict['f1'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[f1_score(y_true, y_pred)],
                                                'upper':[f1_score(y_true_best_scenario, y_pred)],
                                                'lower':[f1_score(y_true_worst_scenario, y_pred)],
                                            })], axis=0)
            
            error_dict['accuracy'] = pd.concat([error_dict['accuracy'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[accuracy_score(y_true, y_pred)],
                                                'upper':[accuracy_score(y_true_best_scenario, y_pred)],
                                                'lower':[accuracy_score(y_true_worst_scenario, y_pred)],
                                            })], axis=0)
            
            error_dict['specificity'] = pd.concat([error_dict['specificity'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[recall_score(y_true, y_pred, pos_label=0)],
                                                'upper':[recall_score(y_true_best_scenario, y_pred, pos_label=0)],
                                                'lower':[recall_score(y_true_worst_scenario, y_pred, pos_label=0)],
                                            })], axis=0)
            
            error_dict['precision'] = pd.concat([error_dict['precision'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[precision_score(y_true, y_pred, zero_division=0)],
                                                'upper':[precision_score(y_true_best_scenario, y_pred, zero_division=0)],
                                                'lower':[precision_score(y_true_worst_scenario, y_pred, zero_division=0)],
                                            })], axis=0)
            
            error_dict['recall'] = pd.concat([error_dict['recall'], 
                                            pd.DataFrame({
                                                'fraction':[f],
                                                'score':[recall_score(y_true, y_pred)],
                                                'upper':[recall_score(y_true_best_scenario, y_pred)],
                                                'lower':[recall_score(y_true_worst_scenario, y_pred)],
                                            })], axis=0)
            
        # plotting graph
        fig, axes = plt.subplots(figsize = (16,8), ncols = 4, nrows = 2)
        
        # Add title
        fig.suptitle(title, fontsize=16)
        
        plt.style.use('default')
        
        axes_iterator = iter([axes[i, j] for i in range(axes.shape[0]) for j in range(axes.shape[1])])
        
        for metric_name, error_df in error_dict.items():
            
            ax=next(axes_iterator)

            ax.plot('fraction', 'score', data=error_df, label='original value', color='r')

            # Add error bars | errorbar is ignored as it requires +- values for points not existing
            error_df = error_df.reset_index(drop=True)
            ax.vlines(x='fraction', ymin='lower', ymax='upper', data=error_df, color='k', label='_nolegend_')
            ax.hlines(y='upper', xmin=error_df['fraction']-0.02, xmax=error_df['fraction']+0.02, data=error_df, color='k', label='_nolegend_')
            ax.hlines(y='lower', xmin=error_df['fraction']-0.02, xmax=error_df['fraction']+0.02, data=error_df, color='k', label='_nolegend_')
            #ax.plot('fraction', 'upper', data=error_df, marker='_', color='k', markersize=1)
            #ax.plot('fraction', 'lower', data=error_df, marker='_', color='k', markersize=1)
            
            # y_erroorr = np.array([error_df['score']-error_df['lower'], error_df['upper']-error_df['score']])
            # y_err = y_error.clip(min=0) # turn negative to 0
            # ax.errorbar('fraction', 'score', data=error_df,
            #              yerr = y_error,
            #              fmt ='o', capsize=10)
            ax.fill_between('fraction', 'lower', 'upper', data=error_df, alpha=.25)
            ax.set_title(metric_name)
            ax.set_ylabel('Metric score', fontsize=9)
            ax.set_xlabel('Proportion of missing links', fontsize=9)
            ax.legend(loc='best', fontsize=9)
            ax.set_ylim([0,1])
        
        # filename = f'{plots_dir}/best_model_hypothetic_CI_{model_name}'
        # plt.savefig(filename+".pdf", format="pdf")
        # plt.savefig(filename+".png", format="png")

        # Plot
        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

        return None

    def multi_plot(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, group_by='name', top_n_features=15, plots=[], show=True, save=False, save_path=''):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # if frac == 'all':
        #     title = f'{self.model_name}, all fractions'
        # else:
        #     title = f'{clf_name}, {100-frac*100:.0f}% removed links'
        title = f'{self.model_name}'
        
        plots_dict = { # TODO: deal with args
            'confusion_matrix': {'threshold':threshold},
            'grouped_evaluation': {'threshold':threshold, 'group_by':group_by},
            'roc_curve': {'threshold':threshold},
            'pr_curve': {'threshold':threshold},
            'pr_curve_vs_threshold': {'threshold':threshold},
            'pr_curve_vs_threshold_grouped': {'threshold':threshold, 'group_by':group_by},
            'probs_distribution': {},
            'roc_curve_split': {'threshold':threshold},
            'pr_curve_split': {'threshold':threshold},
            'grouped_evaluation_split': {'threshold':threshold, 'group_by':group_by},
            'feature_importance': {'n':top_n_features}
        }

        # Some plots require more than one subplot
        extra_subplots = 0
        if 'probs_distribution' in plots:
            extra_subplots += 1
        
        required_subplots = len(plots) + extra_subplots
        nrows= -(required_subplots // -4)
        fig, axes = plt.subplots(figsize = (24,6*nrows), ncols = min(required_subplots, 4), nrows = nrows)#, layout="constrained")
        fig.suptitle(title, fontsize=16) # Add title

        ax_iter = iter(axes.flat)

        for plot in plots: 
            ax = next(ax_iter) # Get next axes object
            if plot == 'confusion_matrix':
                self.plot_confusion_matrix(X, y_true, threshold=threshold, show=False, ax=ax)
            elif plot == 'grouped_evaluation':
                self.plot_grouped_evaluation(X, y_true, threshold=threshold, group_by = group_by, metrices=['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall', 'mcc'], show=False, ax=ax)
            elif plot == 'grouped_evaluation_split':
                self.plot_grouped_evaluation(X, y_true, split_by='community', threshold=threshold, group_by = group_by, metrices=['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall', 'mcc'], show=False, ax=ax)
            elif plot == 'roc_curve':
                self.plot_roc_curve(X, y_true, threshold=threshold, show=False, ax=ax)
            elif plot == 'pr_curve':
                self.plot_pr_curve(X, y_true, threshold=threshold, show=False, ax=ax)
            elif plot == 'roc_curve_split':
                self.plot_roc_curve(X, y_true, split_by='community', threshold=threshold, show=False, ax=ax)
            elif plot == 'pr_curve_split':
                self.plot_pr_curve(X, y_true, split_by='community', threshold=threshold, show=False, ax=ax)
            elif plot == 'pr_curve_vs_threshold':
                self.plot_pr_vs_threshold(X, y_true, threshold=threshold, show=False, ax=ax)
            elif plot == 'pr_curve_vs_threshold_grouped':
                self.plot_pr_vs_threshold(X, y_true, threshold=threshold, group_by=group_by, show=False, ax=ax)
            elif plot == 'probs_distribution':
                self.plot_probs_distribution(X, y_true, show=False, axes=[ax, next(ax_iter)])
            elif plot == 'feature_importance':
                self.plot_feature_importance(n=top_n_features, show=False, ax=ax)
        
        # if save:
        #     filename = f'{plots_dir}/multi_evaluation_{model_name}'
        #     plt.savefig(filename+".pdf", format="pdf")
        #     plt.savefig(filename+".png", format="png")

        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

        return None
    
    
    def classification_report(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, show=True):

        if y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

        # Get predictions based on probabilities, using the given threshold
        y_pred = (y_proba >= threshold).astype('int') 

        if show:
            print(classification_report(y_true, y_pred))

        return None


# -----------------------------

from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import GroupShuffleSplit

class BaseGroupCV(BaseCrossValidator):

    def __init__(self, group_by='name', stratify_by=None, fractions_col = 'fraction', diff_range=1, fractions_train = None, fractions_test = None, stratify_groups_train=None, stratify_groups_test=None, drop_isolates=False, keep_fractions_train=False, keep_fractions_test=False, trueLinks_id=None, drop_existing_links=False, random_state=0):
        
        self.group_by = group_by
        self.stratify_by = stratify_by
        self.diff_range = diff_range
        self.drop_isolates = drop_isolates
        self.drop_existing_links = drop_existing_links
        
        self.fractions_train = fractions_train
        self.fractions_test = fractions_test
        self.fractions_col = fractions_col

        self.stratify_groups_train = stratify_groups_train
        self.stratify_groups_test = stratify_groups_test

        self.keep_fractions_train = keep_fractions_train
        self.keep_fractions_test = keep_fractions_test

        self.trueLinks_id = trueLinks_id
        self.train_link_id = {}
        self.test_link_id = {}

        self.old_indices_mapping = None # save old index for later, as scikit-learn's KFold iterators reset the indices for some reason
        self.random_state = random_state

    def _preprocess(self, X, y=None):
        
        X = X.copy()
        
        # Split to X and y
        if y is None:
            X, y = X.iloc[:,:-1], X.iloc[:,-1]
        else:
            y = y.copy()
        
        # Deal with existing-link instances
        if self.drop_existing_links:
            if self.trueLinks_id is None:
                self.trueLinks_id = X[y == 1]['link_ID'] # Keep original existing links instances for later
            # y[y == -1] = 1 # Relabel subsampled links as existing-links

        # Get groups
        groups = np.array(X[self.group_by].values) # using name -> all reps & fracs & layers of same networks will be in the same groups

        return X, y, groups
    
    def _yield_indices(self, X, y, train_idx, test_idx):
        '''
        A function which further proccess the indices of the train and test sets, and yields the indices of the train and test sets.
        '''
        
        # Spliting the data to test set and train set
        X_train, y_train = X.loc[train_idx], y.loc[train_idx]
        X_test, y_test = X.loc[test_idx], y.loc[test_idx]

        # Keep only desired fraction at each dataset
        if self.keep_fractions_train == False:
            X_train, y_train = X_train[X_train[self.fractions_col].isin(self.fractions_train)], y_train[X_train[self.fractions_col].isin(self.fractions_train)]
        if self.keep_fractions_test == False:
            X_test, y_test = X_test[X_test[self.fractions_col].isin(self.fractions_test)], y_test[X_test[self.fractions_col].isin(self.fractions_test)]

        # Drop isolates
        if self.drop_isolates:
            X_train, y_train = X_train[(X_train['isolate_LL'] != 1) | (X_train['isolate_HL'] != 1)], y_train[(X_train['isolate_LL'] != 1) | (X_train['isolate_HL'] != 1)]
        
        # Undersample
        if self.diff_range != None:
            X_train, y_train = undersample(X_train, y_train, self.diff_range, group_by='subsample_ID')

        if self.drop_existing_links:
            # Drop existing-links in test data | they should not get evaluated (unless test_set = 'true-network')
            bool_remove = X_test['link_ID'].isin(self.trueLinks_id)
            X_test, y_test = X_test[~bool_remove], y_test[~bool_remove]

        # Save link_ids
        self._save_link_ids(X_train, X_test)
        
        return X_train.index, X_test.index
    
    def _save_link_ids(self, X_train, X_test):
        i = max(list(self.train_link_id.keys()), default=-1)+1
        self.train_link_id[i], self.test_link_id[i] = X_train['link_ID'], X_test['link_ID']

    def get_link_ids(self, i=None):
        if i is None:
            return self.train_link_id, self.test_link_id
        else:
            return self.train_link_id[i], self.test_link_id[i]

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class CustomGroupCV(BaseGroupCV):

    def __init__(self, n_splits=1, stratify_test_only_groups=False, **kwargs):
        super().__init__(**kwargs)
        self.n_splits = n_splits
        self.stratify_test_only_groups = stratify_test_only_groups
    
    def split(self, X, y=None, groups=None, stratify_test_only_groups=False):
        
        X, y, groups_distinct = self._preprocess(X, y)
        
        if self.stratify_by is None:

            group_kfold = GroupKFold(n_splits=self.n_splits)
            for train_idx, test_idx in group_kfold.split(X, y, groups_distinct):
                yield self._yield_indices(X, y, train_idx, test_idx)
        
        else:
            
            # Initialize the test_idx_list which will be used to store the indices of the test set of the groups that are present in the test set only, in case stratify_test_only_groups is True
            test_idx_list = np.array([], dtype='int64')
            
            # Get the all groups (present in the train and test sets)
            groups_to_stratify = set(self.stratify_groups_train + self.stratify_groups_test)

            # Should the test set be also stratified by the stratify_by column, but only for the groups that are present in the test set?
            if self.stratify_test_only_groups:
                groups_not_to_stratify = set(self.stratify_groups_test).difference(set(self.stratify_groups_train))
                groups_to_stratify = groups_to_stratify.difference(groups_not_to_stratify)

                # Add the indices of the test set of the groups that are present in the test set only to the test_idx_list, as they are not stratified
                for g in groups_not_to_stratify:
                    test_idx_list = np.append(test_idx_list, X[X[self.stratify_by] == g].index)
                    
            # Subset the data to contain only the groups that are present in the train set
            X_subset = X[X[self.stratify_by].isin(groups_to_stratify)]

            # Get the stratify_by and group_by columns as arrays of values
            groups_strat = np.array(X_subset[self.stratify_by].values)
            groups_distinct_subset = np.array(X_subset[self.group_by].values)

            # StratifiedGroupKFold is used to ensure that the stratify_by column is stratified, while the group_by column is grouped
            sgkf = StratifiedGroupKFold(n_splits=self.n_splits)

            for train_idx, test_idx in sgkf.split(X_subset, y=groups_strat, groups=groups_distinct_subset):

                # get the real (non-continuous) indices, not the positional ones
                train_idx = X_subset.iloc[train_idx].index
                test_idx = X_subset.iloc[test_idx].index

                yield self._yield_indices(X, y, train_idx, np.concatenate((test_idx, test_idx_list)))
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
    
    def get_link_ids(self, i=None):
        if i is None:
            return self.train_link_id, self.test_link_id
        else:
            return self.train_link_id[i], self.test_link_id[i]

class CustomGroupSplit(BaseGroupCV):

    def __init__(self, train_size=0.7, predefined_split=None, **kwargs):
        super().__init__(**kwargs)
        self.train_size = train_size
        self.predefined_split = predefined_split

    def split(self, X, y=None):
        # TODO: this function can be much more efficient by not using the entire dataframe, but only the necessary columns

        X, y, groups = self._preprocess(X, y)
                
        if self.predefined_split != None:
            train_idx = X[X[self.group_by].isin(self.predefined_split['train'])].index            
            test_idx = X[X[self.group_by].isin(self.predefined_split['test'])].index
            yield self._yield_indices(X, y, train_idx, test_idx)

        elif self.stratify_by is None:
            
            # Generate groups
            gss = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state) # train_size = proportion of the groups

            for train_idx, test_idx in gss.split(X, y, groups):
                yield self._yield_indices(X, y, train_idx, test_idx)
                break # only one split is needed

        else:
            #TODO: consider using StratifiedGroupKFold, like this

            # Set variable for convenience
            strat_col = self.stratify_by

            # Get unique values
            strat_uniques = list(X[strat_col].unique())

            # Verify that the stratify_by column contains more than one unique value
            if len(strat_uniques)<=1:
                raise ValueError('The stratify_by column should contain more than one unique value')
            
            # Generate groups - if multiple communities are present
            train_idx = []
            test_idx = []

            # Get the intersection of the train and test sets
            groups_intersection = set(self.stratify_groups_train).intersection(set(self.stratify_groups_test))

            # Loop through the unique values of the stratify_by column
            for group in strat_uniques:
                
                group_mask = X[strat_col] == group

                if group in groups_intersection:
                    # If a group is destined for both train & test set, then split it by the chosen train/test ratio
                    gss = GroupShuffleSplit(n_splits=1, train_size=self.train_size, random_state=self.random_state)
                    train_test_idx = next(gss.split(X[group_mask], y[group_mask], groups[group_mask]))
                    train_idx += list(X[group_mask].iloc[train_test_idx[0]]['link_ID'])
                    test_idx += list(X[group_mask].iloc[train_test_idx[1]]['link_ID'])

                elif group in self.stratify_groups_train:
                    train_idx += list(X[group_mask]['link_ID'])

                elif group in self.stratify_groups_test:
                    test_idx += list(X[group_mask]['link_ID'])

            train_idx = X[X['link_ID'].isin(train_idx)].index
            test_idx = X[X['link_ID'].isin(test_idx)].index
        
            yield self._yield_indices(X, y, train_idx, test_idx)


# class IQROutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, factor=1.5):
#         self.factor = factor

#     def fit(self, X, y=None):
        
#         self.Q1_ = np.percentile(X, 25, axis=0)
#         self.Q3_ = np.percentile(X, 75, axis=0)
#         self.IQR_ = self.Q3_ - self.Q1_
#         return self

#     def transform(self, X, y=None):
#         return X[(X >= (self.Q1_ - self.factor * self.IQR_)).all(axis=1) & 
#                   (X <= (self.Q3_ + self.factor * self.IQR_)).all(axis=1), :]

# class IsoForestOutlierRemover(BaseEstimator, TransformerMixin):
#     def __init__(self, contamination=0.01):
#         self.contamination = contamination

#     def fit(self, X, y=None):
#         self.ifo = IsolationForest(contamination=self.contamination)
#         self.ifo.fit(X)
#         self.inlier_mask_ = self.ifo.predict(X) > 0
#         self.inlier_mean_ = X[self.inlier_mask_].mean(axis=0)
#         return self

#     def transform(self, X, y=None):
#         X_transformed = X.copy()
#         outlier_mask = self.ifo.predict(X) <= 0
#         X_transformed[outlier_mask] = self.inlier_mean_
#         return X_transformed

# def remove_outliers(df):

#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     preds = IsolationForest(contamination=0.005, n_jobs=-1).fit_predict(df[numeric_cols])
#     return df[preds == 1]


def undersample(X, y, diff_range=1, group_by=None, weights_by=None, return_removed=False):
    
    # note: use 'weights' arg of pd.sample() with shortest path..
    #weights_by = 'shortest_path_length'
    
    minor_class = 1
    major_class = 0
    
    keep_idx = [] # list of indices to keep
    
    Xy = pd.concat([X, y], axis=1)
    
    subset_columns = ['repetition', 'class', weights_by] if weights_by is not None else ['repetition', 'class']
    
    # keep the major_class of only the first repetition
    # remove_idx = Xy[(Xy['repetition']>1) & (Xy['class']==major_class)].index
    # Xy = Xy.drop(remove_idx, axis=0)

    if group_by is not None:
        gb = Xy.groupby(group_by)

        for group in gb.groups:

            subset = gb.get_group(group)[subset_columns]
            
            # if subset['repetition'].iloc[0] > 1:
                
            #     keep_idx += subset[subset['class'] == minor_class].index.tolist()
            #     continue

            # Get the sizes of the minor and major classes
            minor_class_count = (subset['class'] == minor_class).sum()
            major_class_count = (subset['class'] == major_class).sum()

            # Determine the sample size (out of the major class dataframe)
            sample_size = min(int(minor_class_count*diff_range), major_class_count) # sample size cannot be larger than class size

            # Get a major class df as a subset of subsample df
            major_class_df = subset[subset['class'] == major_class]

            # Random under-sampling, keep the indices
            keep_idx += major_class_df.sample(n = sample_size, weights = weights_by).index.tolist()
            keep_idx += subset[subset['class'] == minor_class].index.tolist() # keep also the minor class indices
    
    else: # might consider to transform to function, as the code is the same as above
        
        subset = Xy[subset_columns] # faster than using whole df
        
        # Get the sizes of the minor and major classes
        minor_class_count = (subset['class'] == minor_class).sum()
        major_class_count = (subset['class'] == major_class).sum()
        
        # Determine the sample size (out of the major class dataframe)
        sumple_size = min(int(minor_class_count*diff_range), major_class_count) # sample size cannot be larger than class size
        
        # Get a major class df as a subset of subsample df
        major_class_df = subset[subset['class'] == major_class]

        # Random under-sampling, keep the indices
        keep_idx += major_class_df.sample(n = sumple_size, weights = weights_by).index.tolist()
        keep_idx += subset[subset['class'] == minor_class].index.tolist() # keep also the minor class indices
    
    # Combine
    Xy_under = Xy[Xy.index.isin(keep_idx)]
    Xy_under = shuffle(Xy_under)

    # Split
    X_under = Xy_under[X.columns]
    y_under = Xy_under.iloc[:,-1]
    
    if return_removed:
        return X_under, y_under, Xy[~Xy.index.isin(keep_idx)]
    
    else:
        return X_under, y_under


