This directory is described by the following files tree.
The folders and code files are ordered according to the excecution steps.

- 📁 **results**
   - 📄 results_preprocess.Rmd`        `← Loading and process the results data, so each fig will have its own prepared dataset.
   - 📄 results_figs.Rmd`               `← Loading the output of results_preprocess.Rmd and generate figures
   - 📁 **raw**`                     `**←** Contains the raw results, which are mainly the output of the ML pipeline
     -  📄 results_domains.csv`                     `← Contains results from a ML model trained and tested on varying groups (network domains/communities) combinations to assess cross-group generalization.
     -  📄 results_models.csv`                      `← Contains results from different ML models
     -  📄 results_fractions.csv`                     `← Contains results from different fractions rates
     -  📄 train_link_id.csv`                         `← IDs of link in train set (ecological)
     -  📄 test_link_id.csv`                          `← IDs of link in test set (ecological)
     -  📄 train_link_id_domains.csv`                `← IDs of link in train set (non-ecological)
     -  📄 test_link_id_domains.csv`                 `← IDs of link in test set (non-ecological)
      - 📄 feature_importance.csv`                  `← Feature importances of all ML models
      
   - 📁 **intermediate**`             `**←** Contains intermediate processed fils, mainly the output of results_preproccessing
     -  📄 confusion_matrix.csv`                     `← Result of confusion matrix
     -  📄 df_pred_heatmap.csv`                     `← Result of a specific network in the test set, intended for demonstration figure. 
     -  📄 metrics_df_long.csv`                      `← Evaluation metrics of each network, long format
     -  📄 metrics_multi_df_long.csv`                `← Evaluation metrics of each network with multiple models, long format
     -  📄 metrics_type_df_long.csv`                 `← Evaluation metrics of each network with varying group, long format
     -  📄 metrics_fractions_df_long.csv`            `← Evaluation metrics of each network with multiple fractions, long format
     -  📄 network_lvl_features.csv`                 `← Features (network level only) for EDA
     -  📄 pr_auc_values.csv`                        `← PR AUC values for each network (+ no skill calculation)
     -  📄 pr_df.csv`                                `← Results of precision-recall curve
     -  📄 roc_auc_values.csv`                       `← ROC AUC values for each network
     -  📄 roc_df.csv`                               `← Results of roc curve
     -  📄 test_data.csv`                            `← Test set(link ids in test set) with metadata
     -  📄 test_networks.csv`                        `← Networks composing the test set
     -  📄 train_networks.csv`                       `← Networks composing the train set
     
   - 📁 **final**`                     `**←** Contains the final figures and table, mainly the output of results_figs
     -  📄 communities.pdf`                        `← Distributions of performance measures - by community
     -  📄 eval_all.pdf`                             `← Distributions of performance measures
     -  📄 features.csv`                            `← Information about each feature
     -  📄 importance_pres.pdf`                    `← Feature importance for tested ML model (RandomForest)
     -  📄 kruskal_wallis.csv`                       `← Results of Kruskal Wallis test, comparing metrics of different communities
     -  📄 mann_whitney.csv`                       `← Results of Mann-Whitney U Tests comparing the distributions of some metrics for various training and test combinations
     -  📄 networks_table.csv`                      `← Information (source) about each network
     -  📄 networks_summary_properties.csv`        `← Summary of network properties
     -  📄 predictions.pdf`                          `← Link prediction example for a host-parasite network
     -  📄 ROC.pdf`                                `← ROC curve + PR curve
     -  📄 split_set.pdf`                            `← Link prediction within and between community types
     -  📄 SI_community.pdf`                       `← Distribution of link probabilities across different ecological communities
     -  📄 ~~SI_complete~~`                            `← Comparing learning from complete vs subsampled networks
     -  📄 SI_features_hist.pdf`                     `← Histogram of selected network properties
     -  📄 SI_importance.pdf`                      `← Feature importance for all tested ML models
     -  📄 SI_models.pdf`                          `← ML models performance comparison, multiple evaluation metrics
     -  📄 SI_probabilities.pdf`                     `← Distribution of link probabilities obtained from the model
     -  📄 SI_sensitivity`                           `← Comparing performance for different fraction of removed linked
     -  📄 SI_sensitivity_com`                      `← Comparing performance for different fraction of removed linked, for each community
     -  📄 SI_tradeoff.pdf`                          `← The precision-recall tradeoff as a function of classification threshold

Additional datasets used by this piepline are found at:

- 📁 link-predict`                    `← root folder
   - 📁 data
      - 📁 processed
         - 📁 features
            - 📄 features_py.csv`           `← features generated by python script
            - 📄 features_R.csv`            `← features generated by R script
         - 📁 networks
            - 📄 subsamples_edge_lists.csv` `← sub-sampled networks (inc original networks)
            - 📄 subsamples_metadata.csv` `← sub-sampled networks metadata
