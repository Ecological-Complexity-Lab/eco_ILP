import os
import joblib
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score#, top_k_accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, log_loss, brier_score_loss, matthews_corrcoef
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, make_scorer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.inspection import permutation_importance

# Pipeline
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# models
import xgboost as xgb
# import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier

# cross-validation
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# -----------------------------
import warnings

import warnings # TODO: check if this code can be here or in the main file
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning) # turn off some annoying warnings (LogisticRegression's convergence warning)

# from helper.custom_scorer import custom_scorer

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
        'LogisticRegression':LogisticRegression(max_iter=300, n_jobs=-1), # Logistic Regression
        'RandomForestClassifier':RandomForestClassifier(n_jobs=-1), # Random Forest 
        # 'KNeighborsClassifier':KNeighborsClassifier(n_jobs=-1), # KNN
        'XGBClassifier':xgb.XGBClassifier(n_jobs=-1), # XGBoost
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
        
        other_params = {
            'feature_selector': {
                'k': [10, 20, 30, 40, 50, 70]
            },
        }
        
        params_dist = {
            'classifiers':ml_params[classifier_name],
            'feature_selector':other_params['feature_selector']
        }

        return params_dist
    
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

    def fit(self, X_train=None, y_train=None, df=None, cv=None, cv_inner=None, cv_outer=None, scorer_metric='f1', class_weight='balanced', select=None, tuner_name=None, columns_to_ignore=[], n_features=20):

        ## Initial checks
        
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
                               ("feature_selector", SelectKBest(mutual_info_classif, k=n_features)),
                               ("classifier", self.model)])
        
        # Choose a scorer
        scorer = 'f1'
        # scorer = make_scorer(self.custom_scorer, metric=scorer_metric, greater_is_better=True, needs_proba=False) # , exclude_mask = TrueLinks_inx
        
        # Set the parameters of the model and CV tuner
        if isinstance(self.params_dist['classifiers'], dict):
            params_dist = {f'classifier__{k}': v for k, v in self.params_dist['classifiers'].items()} # Rename each key in the dictionary to 'classifier__' + key
        elif isinstance(self.params_dist['classifiers'], list):
            params_dist = {f'classifier__{k}': v for params_dict in self.params_dist['classifiers'] for k, v in params_dict.items()}
        
        params_dist.update({f'feature_selector__{k}': v for k, v in self.params_dist['feature_selector'].items()})

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

    def get_true_classes(self, df, dataset_link_id):
        subset_idx = df['link_ID'].isin(dataset_link_id)
        y_subset = df[subset_idx].iloc[:,-1:]['class'].copy()
        return y_subset
    
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
            trained_model = trained_model.best_estimator_#._final_estimator # check
        elif self.trained_model.__class__.__name__ == 'Pipeline':
            trained_model = trained_model#._final_estimator

        fitted_preprocessor = self.get_fitted_preprocessor() # Get fitted preprocessor
        
        classifier = trained_model.named_steps['classifier']

        # Get feature names
        feature_names = fitted_preprocessor.get_feature_names_out() 

        # Get selected feature names, if feature selection was used in the pipeline
        if 'feature_selector' in trained_model.named_steps:
            feature_selector = trained_model.named_steps['feature_selector']
            selected_features_mask = feature_selector.get_support()
            feature_names = feature_names[selected_features_mask]

        if n is None:
            n = len(feature_names)
        
        if self.model_name == 'LogisticRegression':
            # importance  = trained_model.coef_[0]
            importance  = classifier.coef_[0]
            feature_importance = pd.Series(importance, index=feature_names)
            feature_importance = feature_importance[feature_importance.abs().nlargest(n).index].sort_values()

        else:
            # importance  = trained_model.feature_importances_
            importance  = classifier.feature_importances_
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

        if feature_importance.empty:
            print("No feature importance available.")
            return None

        if n is None:
            n=self.get_fitted_preprocessor().get_feature_names_out().shape[0]

        # Get trained model
        trained_model = self.trained_model 
        if trained_model.__class__.__name__ == 'Pipeline':
            trained_model = trained_model.named_steps['classifier']

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
        bars = ax.barh(y_pos, feature_importance.values, align='center')

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
        y_proba = self.trained_model.predict_proba(X)[:,1]

        return y_proba
    
    def available_metrics(self):
        return list(self.metrics_functions.keys())

    def evaluate(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss']):

        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 

        # Get predictions based on probabilities, using the given threshold
        y_pred = (y_proba >= threshold).astype('int') 
        #y_pred = pipe.predict(X) # not always working
        
        results = {}

        for metric in metrices:
            if metric in self.metrics_functions:
                fun = self.metrics_functions[metric]
                results[metric] = fun(y_true, y_pred, y_proba)
            else:
                print('Unknown metric: ' + metric)           

        # Convert scalar values to single-item lists
        results = {k: [v] for k, v in results.items()}
        
        return pd.DataFrame.from_dict(results)#.round({'f1_score': 2, 'precision': 2,  'recall': 2, 'accuracy': 2, 'ROC_AUC': 2, 'PR_AUC': 2, 'average_precision': 2, 'f1_macro': 2, 'f1_micro': 2, 'f1_weighted': 2 })
    
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
    
    def plot_single_evaluation(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, metrices=['f1', 'precision', 'recall', 'specificity', 'accuracy', 'roc_auc', 'pr_auc', 'average_precision', 'f1_macro', 'f1_micro', 'f1_weighted', 'log_loss'], ax=None, title='Evaluation Metrics', show=True, save=False, save_path=None):
                
            if X is None or y_true is None:
                X, y_true = self.subset_data(df, dataset_link_id)
    
            results = self.evaluate(X, y_true, threshold=threshold, metrices=metrices)

            # convert to wide format
            results = results.T.reset_index().rename(columns={'index': 'metric', 0: 'value'})
    
            # Create figure
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 8))
    
            # Plot
            # results.plot(x='metric', y='value', kind='bar', ax=ax, title=title)

            # same as above but with sns
            sns.barplot(x='metric', y='value', data=results, ax=ax).set_title(title)
    
            # Plot
            if show:
                plt.tight_layout()
                plt.show()
                plt.close()
    
            return results

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
    
    def plot_density_probs(self, X=None, y_true=None, df=None, dataset_link_id=None, threshold=0.5, group_by = None, axes=None, title='', show=True, save=False, save_path=None):
        
        if X is None or y_true is None:
            X, y_true = self.subset_data(df, dataset_link_id)

        # Get probabilities
        y_proba = self.predict_proba(X) 
        
        if axes is None: # if ax is not a part of a subplot
            fig, ax = plt.subplots(figsize=(6,6))

        # Create density plot
        sns.set(style="whitegrid")

        if group_by is None:
            # Separate the predicted probabilities based on the actual class labels
            y_proba_positive = y_proba[y_true == 1]
            y_proba_negative = y_proba[y_true == 0]

            sns.kdeplot(y_proba_positive, bw_adjust=0.5, fill=True, label="Actual: Positive", ax=ax)
            sns.kdeplot(y_proba_negative, bw_adjust=0.5, fill=True, label="Actual: Negative", ax=ax)

            # Calculate percentages for false negatives and false positives
            # false_negatives = np.sum(y_proba_positive < threshold) / len(y_proba_positive) * 100
            # false_positives = np.sum(y_proba_negative >= threshold) / len(y_proba_negative) * 100

            # Add text labels for false negatives and false positives
            # ax.text(0.2, 0.1, f'False Negatives: {false_negatives:.2f}%', color='blue')
            # ax.text(0.6, 0.1, f'False Positives: {false_positives:.2f}%', color='orange')
        else:
            unique_groups = X[group_by].unique()
            for group in unique_groups:
                mask_group = X[group_by] == group
                sns.kdeplot(y_proba[mask_group & (y_true == 1)], bw_adjust=0.5, fill=True, label=f"{group} - Actual: Positive", ax=ax)
                sns.kdeplot(y_proba[mask_group & (y_true == 0)], bw_adjust=0.5, fill=True, label=f"{group} - Actual: Negative", ax=ax)

        ax.axvline(threshold, linestyle='--', label=f'Threshold {threshold}')
        ax.set_xlabel('Predicted Probability of Being Positive')
        ax.set_ylabel('Frequency')
        ax.set_title('Density Plot of Predicted Probabilities by Actual Class')
        ax.legend()
        
        if show:
            plt.tight_layout()
            plt.show()
            plt.close()

    def check_overfitting(self, X_train=None, y_train=None, X_test=None, y_test=None, threshold=0.5, group_by = None, metrices = ['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall']):

        # Get probabilities
        # y_train_proba = self.predict_proba(X_train)
        # y_test_proba = self.predict_proba(X_test) 
        group_by = 'dataset_type'
        X_train[group_by] = 'train'
        X_test[group_by] = 'test'

        X = pd.concat([X_train, X_test], axis=0)
        y_true = pd.concat([y_train, y_test], axis=0)

        self.plot_grouped_evaluation(X, y_true, split_by=group_by, threshold=threshold, group_by = 'name', 
                                metrices=metrices)

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
            elif plot == 'single_evaluation':
                self.plot_single_evaluation(X, y_true, threshold=threshold, metrices=['roc_auc', 'pr_auc', 'f1', 'accuracy', 'specificity', 'precision', 'recall', 'mcc'], show=False, ax=ax)
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
    
    def export_model(self, path='models/fitted_model'):

        model_data = vars(self)
        model_data = {key: value for key, value in model_data.items() if key in ['trained_model']}

        if not path.endswith('.joblib'):
            path = path+'.joblib'

        # Save the model as a joblib file
        joblib.dump(model_data, path)

        return None
    
    def import_model(self, path='models/fitted_model'):
        
        if not path.endswith('.joblib'):
            path = path+'.joblib'

        # Load the model from a joblib file
        model_data = joblib.load(path)

        for key, value in model_data.items():
            setattr(self, key, value)

        return None
        
    
    def export_results(self):

        return None