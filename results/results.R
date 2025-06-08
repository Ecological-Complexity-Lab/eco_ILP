# Import libraries ----------------------------------------------------------
library(tidyverse)
library(ggpubr)
library(ggtext)
library(magrittr)
library(pROC)
library(PRROC)
library(dplyr)
library(reshape2)
library(stringr)
library(RColorBrewer)
library(cowplot)
library(grid)     # For viewport
library(dunn.test)


# Initialize --------------------------------------------------------------

# Define a color palette 

communities <- tibble(community=c('All','Plant-Pollinator','Host-Parasite','Plant-Seed Dispersers','Plant-Herbivore'),
                      community_abbr=c('All','PP','HP','PSD','PH'),
                      community_color=c('orange','#fee190','#e2699f','#b8bdee','#ca97df'),
                      community_color_dark=c('orange','#feda75','#d62976','#659AC6','#962fbf'))
  
  
  
metrics <- tibble(metric=c('balanced_accuracy','recall','precision','f1','mcc','specificity', 'roc_auc', 'pr_auc'),
                  metric_label=c('BA', 'Recall','Precision','F1','MCC','Specificity', 'ROC AUC', 'PR AUC'),
                  metric_color=c('#F8766D','#00B6EB','#00C094','#FB61D7','orange','#A58AFF','#C49A00','#53B400'))
                  

models <- tibble(model=c('Random Forest (same type)','Random Forest (all types)','ML_single','Connectance','Matching_Centrality','SBM','Ensamble'),
                 label=c('ILP (same)','ILP','TML','C','MC','SBM','Ensemble'),
                 color=c('orange','orange','#861ea5','gray20','gray40','gray60','pink'))

# Set a threshold
threshold <- 0.5 

# Choose whether to export figures and tables
export = F

export_fig <- function(fig, to_file, w=10, h=5){
  pdf(paste(paper_figs_path, to_file, sep = ""), w, h)
  print(fig)
  dev.off()
}

# Folders
path_main_dir = "/Users/shai/GitHub/ecomplab/link-predict/" # Path of the main folder
path_metadata = paste0(path_main_dir, "data/processed/") # Path of proccessed networks & features data, serving as complementary metadata
path_raw_results = paste0(path_main_dir, "results/raw/") # Path of raw results
path_intermediate_results = paste0(path_main_dir, "results/intermediate/") # Path of intermediate results, output of this script
path_final_results = paste0(path_main_dir, "results/final/") # Path of final results, output of this script
paper_figs_path <- '/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/images/' # Path of final results, output of this script

# Define the plots' theme
fontsize_main_NEE <- 7 # The size of the font is relative to the size of the figure. You can change it here.
paper_figs_theme_main_NEE <-
  theme_bw()+
  theme(panel.grid = element_blank(),
        panel.border = element_rect(color = "black",fill = NA,linewidth = 1),
        panel.spacing = unit(0.5, "cm", data = NULL),
        axis.text = element_text(size=fontsize_main_NEE, color='black'),
        axis.title = element_text(size=fontsize_main_NEE, color='black'),
        axis.line = element_blank(),
        legend.text=element_text(size=fontsize_main_NEE, color='black'),
        legend.title=element_text(size=fontsize_main_NEE, color='black'))

fontsize_SI_NEE <- 14 # The size of the font is relative to the size of the figure. You can change it here.
paper_figs_theme_SI_NEE <-
  theme_bw()+
  theme(panel.grid = element_blank(),
        panel.border = element_rect(color = "black",fill = NA,linewidth = 1),
        panel.spacing = unit(0.5, "cm", data = NULL),
        axis.text = element_text(size=fontsize_SI_NEE, color='black'),
        axis.title = element_text(size=fontsize_SI_NEE, color='black'),
        axis.line = element_blank(),
        legend.text=element_text(size=fontsize_SI_NEE, color='black'),
        legend.title=element_text(size=fontsize_SI_NEE, color='black'))

# Define functions ---------------------------------------------------------
# Define a function to summarize and compute metrics
calc_conf_metrics <- function(.data) {
  .data %>%
    summarize(
      TP = sum(category == "TP"),
      FP = sum(category == "FP"),
      TN = sum(category == "TN"),
      FN = sum(category == "FN"),
      .groups = "drop"
    ) %>%
    mutate(
      accuracy = (TP + TN) / (TP + TN + FP + FN),
      precision = TP / (TP + FP),
      recall = TP / (TP + FN),
      specificity = TN / (TN + FP), 
      balanced_accuracy = 0.5 * ((TP / (TP + FN)) + (TN / (TN + FP))),
      f1 = 2 * (precision * recall) / (precision + recall)
    )
}

# Exploratory Data Analysis ------------------------------------------------

## Filtered networks --------------------------------------------------------

filtered_networks <- read_csv(paste0(path_raw_results, "filtered_networks.csv"))
nrow(filtered_networks)
names(filtered_networks)
filtered_networks %>% 
  ggplot(aes(x=net_size, y=connectance, color=community)) +
  geom_point() +
  geom_hline(yintercept = 0.1, linetype = "dashed") +
  scale_color_manual(values = communities$community_color_dark) +
  labs(x = "Network size", y = "Connectance") +
  paper_figs_theme_SI_NEE



## Features distributions ---------------------------------------------------

# Load the features dataframe
network_lvl_df <- read_csv(paste0(path_intermediate_results, "network_lvl_df.csv"))
nrow(network_lvl_df)

SI_network_properties_plot <- 
  network_lvl_df %>%
  select(
    community, 
    `Network size` = network_size, 
    `Number of links` = interactions_count, 
    Connectance = connectance, 
    `Number of links per species` = links.per.species
  ) %>%
  pivot_longer(
    cols = c(`Network size`, `Number of links`, Connectance, `Number of links per species`), 
    names_to = "feature", 
    values_to = "value"
  ) %>% 
  ggplot(aes(x = value, fill = feature)) +
    geom_histogram(position = "dodge") +
    facet_wrap(~feature, scales = "free") +
    scale_fill_discrete() +
    labs(x = "Value", y = "Count", fill='Property')+
    paper_figs_theme_SI_NEE
  

SI_network_properties_plot
export_fig(SI_network_properties_plot, 'SI_network_properties_plot.pdf', 10, 5)

filter_list <- c("Plant-Pollinator", "Host-Parasite", "Plant-Seed Dispersers", "Plant-Herbivore")

# Long format features daatframe
network_lvl_df_long <- 
  network_lvl_df %>%
  filter(community %in% filter_list) %>%
  select(community, network_size, interactions_count, connectance, links.per.species) %>%
  rename('average degree' = links.per.species) %>%
  pivot_longer(cols = c('network_size', 'interactions_count', 'connectance', 'average degree'), names_to = "feature", values_to = "value") #, -community

# Create a histogram for each feature
ggplot(network_lvl_df_long, aes(x = value, fill = community)) +
  geom_histogram(position = "dodge") +
  facet_grid(community ~ feature, scales = "free") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  # Take colors from communities
  scale_fill_manual(values = setNames(communities$community_color_dark, communities$community)) +
  labs(x = "Value", y = "Count")



# PCA for network similarity ------------------------------------------------
# Load the dataframe
pca_df <- 
  read_csv(paste0(path_intermediate_results, "pca_df.csv"))

# Plot
SI_networks_PCA <- 
  ggplot(pca_df, aes(x = PC2, y = PC1, color = community)) +
  geom_point(size=2) +
  labs(x = "PC1", y = "PC2", color='Community') +
  scale_color_manual(values = setNames(communities$community_color_dark, communities$community)) +
  paper_figs_theme_SI_NEE +
  theme(legend.position = "inside",
        legend.position.inside = c(0.75,0.85))

SI_networks_PCA
export_fig(SI_networks_PCA, 'SI_networks_PCA.pdf', 6, 6)

# Feature correlations ------------------------------------------------------

feature_correlations <- function(data, plot_corr, title = "Clustered Correlation Matrix") {
  # 1. Select only numeric columns and remove constant ones
  data_clean <- data %>% 
    select_if(is.numeric) %>% 
    select_if(~ n_distinct(.) > 1)
  
  # 2. Compute the correlation matrix, using pairwise.complete.obs to handle NAs, and take absolute values
  corr_matrix <- cor(data_clean, use = "pairwise.complete.obs") %>% abs()
  
  # Check if the correlation matrix has any NA/NaN/Inf values
  if (any(!is.finite(corr_matrix))) {
    stop("The correlation matrix contains non-finite values. Please check your data.")
  }
  
  # 3. Perform hierarchical clustering on the correlation matrix.
  diss <- as.dist(1 - corr_matrix)
  hc <- hclust(diss, method = "complete")
  
  # 4. Extract the order of features from the clustering
  ordered_features <- rownames(corr_matrix)[hc$order]
  
  # 5. Convert the correlation matrix to long format and keep only the lower triangle
  correlation_df <- as_tibble(corr_matrix, rownames = "feature1") %>% 
    pivot_longer(
      cols = -feature1,
      names_to = "feature2",
      values_to = "corr"
    ) %>%
    filter(feature1 != feature2) %>%
    mutate(
      feature1 = factor(feature1, levels = ordered_features),
      feature2 = factor(feature2, levels = ordered_features)
    ) %>% 
    filter(as.numeric(feature1) > as.numeric(feature2))
  
  if (plot_corr){
    # 6. Plot the clustered correlation heatmap using ggplot2
    p <- ggplot(correlation_df, aes(feature1, feature2, fill = corr)) +
      geom_tile(color = "white") +
      scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      labs(
        title = title,
        x = "", y = "", fill = "Correlation"
      ) +
      paper_figs_theme_SI_NEE +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
    return(p)
  } else {
    return(correlation_df)
  }
}


## Correlations of network-level properties -----------------------------------
network_lvl_df <- read_csv(paste0(path_intermediate_results, "network_lvl_df.csv")) %>% 
  select(-community)
SI_correlations_network_lvl <- feature_correlations(network_lvl_df, plot_corr = T,
                                                    title = "Correlations of network-level properties")
SI_correlations_network_lvl
export_fig(SI_correlations_network_lvl, 'SI_correlations_network_lvl.pdf', 10, 10)

## Correlations of node-level properties --------------------------------------
node_lower_lvl_df <- read_csv(paste0(path_intermediate_results, "node_lower_lvl_df.csv"))
SI_correlations_node_lower_lvl <- feature_correlations(node_lower_lvl_df, plot_corr = T,
                                                       title = "Correlations of lower node-level properties")
SI_correlations_node_lower_lvl
export_fig(SI_correlations_node_lower_lvl, 'SI_correlations_node_lower_lvl.pdf', 10, 10)

node_higher_lvl_df <- read_csv(paste0(path_intermediate_results, "node_higher_lvl_df.csv"))
SI_correlations_node_higher_lvl <- feature_correlations(node_higher_lvl_df, plot_corr = T,
                                                        title = "Correlations of higher node-level properties")
SI_correlations_node_higher_lvl
export_fig(SI_correlations_node_higher_lvl, 'SI_correlations_node_higher_lvl.pdf', 10, 10)

## Correlations of link-level properties --------------------------------------
link_lvl_df <- read_csv(paste0(path_intermediate_results, "link_lvl_df.csv"))
SI_correlations_link_lvl <- feature_correlations(link_lvl_df, plot_corr = T, title = "Correlations of link-level properties")
SI_correlations_link_lvl
export_fig(SI_correlations_link_lvl, 'SI_correlations_link_lvl.pdf', 10, 10)

## Correlation histograms ------------------------------------------------------
network_lvl_correlation_df <- feature_correlations(network_lvl_df, plot_corr = F)
node_lower_lvl_correlation_df <- feature_correlations(node_lower_lvl_df, plot_corr = F)
node_higher_lvl_correlation_df <- feature_correlations(node_higher_lvl_df, plot_corr = F)
link_lvl_correlation_df <- feature_correlations(link_lvl_df, plot_corr = F)


SI_feature_correlations <- 
  bind_rows(
    network_lvl_correlation_df %>% mutate(level = 'Network'),
    node_lower_lvl_correlation_df %>% mutate(level = 'Node lower'),
    node_higher_lvl_correlation_df %>% mutate(level = 'Node higher'),
    link_lvl_correlation_df %>% mutate(level = 'Link')
  ) %>%
  mutate(level = factor(level, levels = c('Node lower', 'Node higher', 'Link','Network'))) %>% 
  ggplot(aes(x=corr, fill=level)) +
  geom_histogram()+
  facet_wrap(~level, scales = "free_y") +
  geom_vline(xintercept = 0.8, linetype = "dashed", color = "red") +
  labs(x = 'Correlation value', y = 'Count') +
  paper_figs_theme_SI_NEE+
  theme(legend.position = 'none')

SI_feature_correlations
export_fig(SI_feature_correlations, 'SI_feature_correlations.pdf', 10, 5)

# Compare ILP to TLP ------------------------------------------------------

# Load the dataset
compare_models_metrics_df <- read_csv(paste0(path_intermediate_results, 'compare_other_models_metrics_df.csv'))

# Define the desired order of metrics
desired_metrics <- c('Recall', 'Precision', 'F1', 'PR AUC')

# Write source data according to NEE guidelines
compare_models_metrics_df %>% 
  mutate(metric = factor(metric, levels = metrics$metric, labels =  metrics$metric_label)) %>%
  # Filter to keep only the desired metrics
  filter(metric %in% desired_metrics) %>%
  # Reorder the metric factor according to the desired order
  mutate(metric = factor(metric, levels = desired_metrics)) %>%
  mutate(model = factor(model, levels = models$model, labels = models$label)) %>%
  filter(model %in% c('ILP','TML','SBM','MC','C')) %>% 
  write_csv('/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Fig_2_source_data.csv')

ILP_vs_TLP <- 
  compare_models_metrics_df %>% 
  mutate(metric = factor(metric, levels = metrics$metric, labels =  metrics$metric_label)) %>%
  # Filter to keep only the desired metrics
  filter(metric %in% desired_metrics) %>%
  # Reorder the metric factor according to the desired order
  mutate(metric = factor(metric, levels = desired_metrics)) %>%
  mutate(model = factor(model, levels = models$model, labels = models$label)) %>%
  filter(model %in% c('ILP','TML','SBM','MC','C')) %>%
  ggplot(aes(x = metric, y = value, fill = model, color = model)) +
  geom_boxplot(show.legend = T, color = 'black', outlier.alpha=0.6) +
  scale_fill_manual(values = setNames(models$color, models$label), labels=c('ILP','ML (TLP)','SBM (TLP)','MC (TLP)','C (TLP)')) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  labs(y = "Metric value", x = "Metric") +
  paper_figs_theme_main_NEE +
  theme(legend.position = "inside",
        legend.position.inside = c(0.87,0.82),
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank()) # Rotating x-axis labels 45 degrees

ILP_vs_TLP
export_fig(ILP_vs_TLP, 'ILP_vs_TLP.pdf', 7.4, 5) # 7.4 is the 188mm width for NEE

SI_ILP_vs_TLP <-
  compare_models_metrics_df %>% 
  mutate(metric = factor(metric, levels = metrics$metric, labels =  metrics$metric_label)) %>%
  mutate(model = factor(model, levels = models$model, labels = models$label)) %>%
  filter(model %in% c('ILP','TML','SBM','MC','C')) %>%
  ggplot(aes(x = metric, y = value, fill = model, color = model)) +
  geom_boxplot(show.legend = T, color = 'black') +
  scale_fill_manual(values = setNames(models$color, models$label)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25)) +
  labs(y = "Metric value", x = "Metric") +
  paper_figs_theme_SI_NEE +
  theme(legend.position = c(0.92,0.15),
        legend.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title.x = element_blank()) # Rotating x-axis labels 45 degrees


SI_ILP_vs_TLP
export_fig(SI_ILP_vs_TLP, 'SI_ILP_vs_TLP.pdf', 10, 5)


# Feature importance ------------------------------------------------------

feature_importance <- read_csv(paste0(path_raw_results, "feature_importance_nCV.csv"))

# Prepare the dataframe
feature_importance_long <- 
  feature_importance %>%
  filter(model == "RandomForestClassifier") %>%
  group_by(feature, model) %>% # Group by feature - when there are folds
  summarise(importance_mean = mean(importance),
            importance_min = min(importance),
            importance_max = max(importance),
            importance_sd = sd(importance)) %>%
  ungroup() %>%
  top_n(15, abs(importance_mean)) %>% # Select top n features for each model
  arrange(importance_mean)

# Convert feature to factor and specify the levels
feature_importance_long$feature <- factor(feature_importance_long$feature, levels = feature_importance_long$feature)

# Create a feature importance plot
SI_feature_importance <-
  ggplot(feature_importance_long, aes(feature, importance_mean, fill = model)) +
    geom_col(show.legend = FALSE) +
    geom_errorbar(aes(ymin = importance_mean-importance_sd, ymax = importance_mean+importance_sd), width = 0.2) +
    coord_flip() +
    labs(y = "Mean Importance", x = "Feature") + # no need for units in the X axis?
    paper_figs_theme_SI_NEE

SI_feature_importance
export_fig(SI_feature_importance, 'SI_feature_importance.pdf', 10, 5)

## Feature importance trends ------------------------------------------------------

# load rds, csv too big
compare_models_features_long <- readRDS(paste0(path_intermediate_results, "temp_compare_models_features_long.rds"))

# Quick exploration of degree values
compare_models_features_long %>%
  filter(model == "Connectance") %>% 
  filter(feature %in% c("degree_HL", "degree_LL")) %>% 
  filter(value<100) %>%   
  # filter(value > 0) %>%
  group_by(feature) %>%
  summarise(min = min(value), max = max(value), mean = mean(value), sd = sd(value), n=n())

compare_models_features_long %>%  
  filter(model == "Connectance") %>% 
  filter(feature %in% c("degree_HL", "degree_LL")) %>% filter(value==0)
  
# Define the number of bins
num_bins <- 10 

binned <- 
  compare_models_features_long %>%
  filter(feature %in% c("degree_HL", "degree_LL")) %>% 
  filter(value <= 100) %>%
  filter(value>0) %>% 
  group_by(feature) %>%
  group_modify(~ {
    bin_breaks <- quantile(.x$value, probs = seq(0, 1, length.out = num_bins + 1), na.rm = TRUE)
    bin_breaks <- unique(bin_breaks)
    if (.y$feature == "degree_HL") {
      bin_breaks <- c(bin_breaks, 50)
    } else {
      bin_breaks <- c(bin_breaks, 30)
    }
    bin_breaks <- sort(bin_breaks)
    print(paste("Feature:", .y$feature, "- bin_breaks:", toString(bin_breaks)))
    
    temp <- .x %>%
      mutate(feature_bin = cut(value, 
                               breaks = bin_breaks, 
                               include.lowest = TRUE, 
                               ordered_result = TRUE))
    
    print(paste("Ordered feature_bin levels for", .y$feature, ":", toString(levels(temp$feature_bin))))
    
    # Convert to character before returning
    temp %>%
      mutate(feature_bin = as.character(feature_bin))
  }) %>%
  ungroup()

# Next, determine a common ordering for the bins
# We'll extract the lower bound from the bin labels and sort accordingly.
extract_lower_bound <- function(bin_label) {
  # bin_label is something like "[0,1]" or "(1,2]". Extract the first number.
  # Remove leading '[' or '(' and then take everything up to the comma.
  as.numeric(str_extract(bin_label, "(?<=\\[|\\().+?(?=,)"))
}

# Get all unique bin levels as characters from the combined data
all_levels <- unique(as.character(binned$feature_bin))
# Order them by their numeric lower bound
ordered_levels <- all_levels[order(sapply(all_levels, extract_lower_bound))]

# Now reassign feature_bin as an ordered factor with the common levels:
compare_model_features_degree_binned <- binned %>%
  mutate(feature_bin = factor(feature_bin, levels = ordered_levels, ordered = TRUE))

# (Optional) Print the common levels
print(paste("Common ordered levels:", toString(levels(compare_model_features_degree_binned$feature_bin))))



# Compare the number of observations in each bin per model and metric
SI_bins_count <-
compare_model_features_degree_binned %>%
  # All models have the same number of observations so filter just for one of them.
  filter(model == "Connectance") %>% 
  group_by(feature, feature_bin) %>%
  summarise(n = n()) %>%
  ungroup() %>% 
  ggplot(aes(x = feature_bin, y = n, fill = feature)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_grid(~ feature, 
             scales = "free",
             labeller = labeller(
               feature = c(degree_HL = "Degree (high trophic level)", degree_LL = "Degree (low trophic level)")
             )) +
  labs(x = "Degree bin", y = "Number of observations", fill = "Feature") +
  scale_fill_manual(values = c('steelblue', 'darkgreen')) +
  paper_figs_theme_SI_NEE +
  theme(legend.position = "none",
        strip.text = element_text(size = 12, face = "bold"),
        axis.text.x = element_text(angle = 45, hjust = 1, size = 10))
SI_bins_count
export_fig(SI_bins_count, 'SI_bins_count.pdf', 10, 5)

# Calculating the proportion of correct predictions per bin
models_preformance_trend <- 
  compare_model_features_degree_binned %>%
  mutate(model = factor(model, levels = models$model, labels = models$label)) %>%
  rename(category = outcome) %>% # for calc_conf_metrics
  group_by(model, feature, feature_bin) %>%
  calc_conf_metrics() %>%
  ungroup() %>% 
  select(model, feature, feature_bin, recall, precision) %>%
  pivot_longer(
    cols = c("recall", "precision"),
    names_to = "metric",
    values_to = "proportion_correct"
  ) %>% 
  mutate(metric = factor(metric, levels = c("recall", "precision")))

# Write source data according to NEE guidelines
write_csv(models_preformance_trend, '/Users/shai/Dropbox (Personal)/Apps/Overleaf/2024 Link prediction in multiple networks/revision_2/NEE/Fig_3_source_data.csv')

# Create plot
performance_trend <-
  models_preformance_trend %>%
  ggplot(aes(x = feature_bin, y = proportion_correct, color = model, group = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_grid(metric ~ feature, 
             scales = "free",
             labeller = labeller(
               metric = c(f1 = "F1", recall='Recall', precision='Precision'),
               feature = c(degree_HL = "Degree (high trophic level)", degree_LL = "Degree (low trophic level)")
             )) +
  labs(
    x = "Degree value bins",
    y = "Metric value",
    color = "Model"
  ) +
  scale_color_manual(values = setNames(models$color, models$label), labels=c('ILP','ML (TLP)','SBM (TLP)','MC (TLP)','C (TLP)')) +
  scale_x_discrete(labels = function(x) gsub(",", "-", gsub("\\(|\\]|\\[", "", x))) +
  paper_figs_theme_main_NEE +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 7, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10)
  )
performance_trend
export_fig(performance_trend, 'performance_trend.pdf', 18.8/2.54, 6)


# For the SI:
# Calculating the proportion of correct predictions per bin
SI_models_preformance_trend <- 
  compare_model_features_degree_binned %>%
  mutate(model = factor(model, levels = models$model, labels = models$label)) %>%
  rename(category = outcome) %>% # for calc_conf_metrics
  group_by(model, feature, feature_bin) %>%
  calc_conf_metrics() %>%
  ungroup() %>% 
  select(model, feature, feature_bin, f1, balanced_accuracy) %>%
  pivot_longer(
    cols = c("f1", "balanced_accuracy"),
    names_to = "metric",
    values_to = "proportion_correct"
  ) %>% 
  mutate(metric = factor(metric, levels = c("f1", "balanced_accuracy")))

SI_performance_trend <-
  SI_models_preformance_trend %>%
  ggplot(aes(x = feature_bin, y = proportion_correct, color = model, group = model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  facet_grid(metric ~ feature, 
             scales = "free",
             labeller = labeller(
               metric = c(f1 = "F1", balanced_accuracy = "Balanced accuracy", accuracy = "Accuracy", specificity = "Specificity"),
               feature = c(degree_HL = "Degree (high trophic level)", degree_LL = "Degree (low trophic level)")
             )) +
  labs(
    x = "Degree value bins",
    y = "Metric value",
    color = "Model"
  ) +
  scale_color_manual(values = setNames(models$color, models$label), labels=c('ILP','ML (TLP)','SBM (TLP)','MC (TLP)','C (TLP)')) +
  scale_x_discrete(labels = function(x) gsub(",", "-", gsub("\\(|\\]|\\[", "", x))) +
  paper_figs_theme_SI_NEE +
  theme(
    legend.position = "bottom",
    strip.text = element_text(size = 7, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10)
  )
SI_performance_trend
export_fig(SI_performance_trend, 'SI_performance_trend.pdf', 10, 6)


# Per-community evaluations ----------------------------------------------------


## Evaluation metrics per community --------------------
metrics_df_long <- 
  read_csv(paste0(path_intermediate_results, "metrics_df_long.csv")) %>% 
  left_join(metrics, by = "metric") %>%
  # Filter to keep only the desired metrics
  filter(metric_label %in% desired_metrics) %>%
  # Reorder the metric factor according to the desired order
  mutate(metric_label = factor(metric_label, levels = desired_metrics)) %>% 
  left_join(communities, by = "community")

metrics_df_long %>% 
  group_by(fold, community) %>%
  summarise(n=n())

SI_per_community_evaluations <- 
  metrics_df_long %>% 
  ggplot(aes(x = metric_label, y = value, fill = community)) +
  geom_boxplot(notch=T) +
  scale_fill_manual(values = setNames(communities$community_color, communities$community)) +
  labs(y = "Metric value", x = "Metric", fill='Community') +
  paper_figs_theme_SI_NEE + 
  theme(legend.position = 'top',
        axis.title.x = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

SI_per_community_evaluations
export_fig(SI_per_community_evaluations, 'SI_per_community_evaluations.pdf', 10, 5)


## PR curves -------------------------------------------------------------------

# Load the dataframes with fold information
pr_df <- read_csv(paste0(path_intermediate_results, "pr_df.csv")) %>% 
  left_join(communities, by = "community")
auc_df <- read.csv(paste0(path_intermediate_results, "auc_df.csv"))

# Calculate mean AUC for each community
auc_summary <- auc_df %>%
  select(community, PR_auc) %>%
  left_join(communities, by = "community") %>%
  mutate(PR_label= paste0(community_abbr, " (AUC: ", round(PR_auc, 2), ")")) %>% 
  mutate(community_abbr = fct_reorder(community_abbr, PR_auc, .desc = TRUE))

# Write figure source data according to NEE guidelines
pr_df %>% 
  mutate(community_abbr = factor(community_abbr, levels=c("All","HP","PSD", "PH", "PP" ))) %>% 
  write_csv('/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Fig_4a_source_data.csv')

# Panel A of Fig. 4
pr_plot <- 
  pr_df %>% 
  mutate(community_abbr = factor(community_abbr, levels=c("All","HP","PSD", "PH", "PP" ))) %>% 
  ggplot(aes(x = recall, y = precision, color = community_abbr)) +
  geom_line(linewidth=1) +
  # geom_ribbon(aes(ymin = mean_precision - sd_precision, ymax = mean_precision + sd_precision, fill = community), alpha = 0.1) +
  labs(x = "Recall", y = "Precision", color="Community") +
  scale_color_manual(values = setNames(communities$community_color_dark, communities$community_abbr),
                     labels = setNames(auc_summary$PR_label, auc_summary$community_abbr)) +
  paper_figs_theme_main_NEE + 
  theme(legend.position = "inside", 
        legend.position.inside = c(0.6,0.7))
pr_plot

## Kruskal Wallis test - comparing metrics of different communities ---------
library(broom)
library(rstatix)

desired_metrics <- c('Recall', 'Precision', 'Specificity', 'F1', 'BA', 'PR AUC')

kw_comparison <- 
metrics_df_long %>%
  # Group the data by each metric
  group_by(metric_label) %>%
  # Nest the data for each metric into a list-column
  nest() %>%
  # Perform the Kruskal-Wallis test and tidy the results
  mutate(
    kw_test = map(data, ~ kruskal.test(value ~ community_abbr, data = .x)),
    kw_tidy = map(kw_test, tidy)
  ) %>%
  # Unnest the KW test results to have one row per metric
  unnest(kw_tidy) %>%
  # Rename columns for clarity
  rename(
    kw_statistic = statistic,
    kw_parameter = parameter,
    kw_p_value = p.value
  ) %>%
  # Perform Dunn's test for metrics with p.value < 0.05
  mutate(
    dunn_test = if_else(
      kw_p_value < 0.05,
      map(data, ~ dunn_test(.x, value ~ community_abbr, p.adjust.method = "bonferroni")),
      list(NULL)  # Assign NULL for non-significant metrics
    )
  ) %>%
  # Select relevant columns
  select(metric_label, kw_statistic, kw_parameter, kw_p_value, dunn_test) %>%
  # Unnest the Dunn test results for significant metrics
  # This will create multiple rows per metric, one for each community comparison
  unnest(dunn_test, keep_empty = TRUE) %>%
  # Arrange the results for better readability
  arrange(metric_label, group1, group2)

  
# Filter significant Dunn test results
significant_dunn <- kw_comparison %>%
  filter(kw_p_value < 0.05, !is.na(group1)) %>%
  mutate(pair = paste(group1, group2, sep = " vs "))
# Calculate medians per metric and community
medians_df <- metrics_df_long %>%
  group_by(metric_label, community_abbr) %>%
  summarize(median_value = median(value, na.rm = TRUE), .groups = "drop")
# Calculate median differences for significant comparisons
final_results <- kw_comparison %>%
  filter(kw_p_value < 0.05, !is.na(group1)) %>%  # Keep only significant comparisons
  # Join to get median values for group1
  left_join(medians_df, by = c("metric_label", "group1" = "community_abbr")) %>%
  rename(median_group1 = median_value) %>%
  # Join to get median values for group2
  left_join(medians_df, by = c("metric_label", "group2" = "community_abbr")) %>%
  rename(median_group2 = median_value) %>%
  # Calculate the difference in medians
  mutate(median_diff = median_group1 - median_group2)
# Add significance labels
significant_dunn <- final_results %>%
  mutate(
    pair = paste(group1, "vs", group2, sep = " "),
    p.adj.signif = case_when(
      p.adj < 0.001 ~ "***",
      p.adj < 0.01  ~ "**",
      p.adj < 0.05  ~ "*",
      TRUE          ~ "ns"
    )
  )
# Create the enhanced heatmap with median differences and significance labels

# Apply the community mapping to the metrics_df_long tibble
significant_dunn %<>%
  mutate(pair = paste(group1, group2, sep = " vs ")) %>% 
  mutate(label = paste0(p.adj.signif, "\nΔ=", round(median_diff, 2)))


# Write figure source data according to NEE guidelines
significant_dunn %>% 
  filter(metric_label %in% c('Recall', 'Precision', 'F1', 'PR AUC')) %>% 
  write_csv('/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Fig_4b_source_data.csv')

# Panel B of Fig. 4
pairwise_comparisons_plot <-
  significant_dunn %>% 
  filter(metric_label %in% c('Recall', 'Precision', 'F1', 'PR AUC')) %>% 
  ggplot(aes(x = metric_label, y = pair)) +
    geom_tile(aes(fill = median_diff), color = "white") +  # Tiles colored by median_diff
    geom_text(aes(label = p.adj.signif), color = "black", size = 4) +  # Significance labels
  scale_fill_gradient2(
    low = "#1984c5",          # Blue for negative differences
    mid = "white",            # White for zero difference
    high = "#c23728",         # Red for positive differences
    midpoint = 0,             # Midpoint at zero
    limits = c(min(significant_dunn$median_diff, na.rm = TRUE), 
               max(significant_dunn$median_diff, na.rm = TRUE)),  # Ensures the scale covers your data range
    name = "Difference\nin medians"  # Legend title with line break for better readability
  ) +
    paper_figs_theme_main_NEE +
    theme(axis.title = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
      panel.grid = element_blank()
    ) +
    labs(
      x = "Metric",
      y = "Community pair"
    )
pairwise_comparisons_plot


per_community_evaluations <- 
cowplot::plot_grid(pr_plot, pairwise_comparisons_plot, 
                   labels = c("a", "b"), label_size = 7,
                   ncol = 2,
                   align = "h",        # Align horizontally
                   axis = "tb",        # Align top and bottom axes
                   rel_widths = c(1,1.3))  # Ensure both plots have equal widths)
export_fig(per_community_evaluations, 'per_community_evaluations.pdf', 18.8/2.54, 3.5)


# For SI:
SI_pairwise_comparisons_plot <-
  ggplot(significant_dunn, aes(x = metric_label, y = pair)) +
  geom_tile(aes(fill = median_diff), color = "white") +  # Tiles colored by median_diff
  geom_text(aes(label = p.adj.signif), color = "black", size = 4) +  # Significance labels
  scale_fill_gradient2(
    low = "#1984c5",          # Blue for negative differences
    mid = "white",            # White for zero difference
    high = "#c23728",         # Red for positive differences
    midpoint = 0,             # Midpoint at zero
    limits = c(min(significant_dunn$median_diff, na.rm = TRUE), 
               max(significant_dunn$median_diff, na.rm = TRUE)),  # Ensures the scale covers your data range
    name = "Difference\nin medians"  # Legend title with line break for better readability
  ) +
  paper_figs_theme_SI_NEE +
  theme(axis.title = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1),  # Rotate x-axis labels
        panel.grid = element_blank()
  ) +
  labs(
    x = "Metric",
    y = "Community pair"
  )
SI_pairwise_comparisons_plot

# Write the file for the SI
export_fig(SI_pairwise_comparisons_plot, 'SI_pairwise_comparisons_plot.pdf', 6, 5)

# Write results to a table for the manuscript in latex format
significant_dunn %>% 
  mutate(kw_statistic = round(kw_statistic, 2),
         median_group1 = round(median_group1, 2),
         median_group2 = round(median_group2, 2),
         median_diff = round(median_diff, 2),
         p.adj = round(p.adj, 3)) %>% 
  select(Metric = metric_label, 
         `KW statistic` = kw_statistic, 
         `Community 1` = group1, 
         `Community 2` =group2, 
         `Median 1` = median_group1, 
         `Median 2` = median_group2, 
         `P value (adjusted)` = p.adj) %>%
  knitr::kable("latex", booktabs = T, caption = "Kruskal-Wallis and Dunn's test results for pairwise comparisons of metrics between communities.")
  
  
  
# Inter-community comparisons --------------------------------------------------
labels <- c("['Plant-Herbivore']",
            "['Plant-Pollinator']",
            "['Plant-Seed Dispersers', 'Plant-Pollinator', 'Plant-Herbivore', 'Host-Parasite']",
            "['Plant-Seed Dispersers', 'Plant-Herbivore', 'Host-Parasite']", 
            "['Host-Parasite']",
            "['Plant-Seed Dispersers']")

# Remove unwanted characters
clean_labels <- gsub("\\['|'\\]", "", labels)

# Replace level 3 with "all"
clean_labels[3] <- "All"
clean_labels[4] <- "No PP"

create_heatmap_plot <- function(data, filter_condition = NULL) {
  # Apply the filter condition if provided
  if (!is.null(filter_condition)) {
    data <- data %>% filter(!!rlang::parse_expr(filter_condition))
  }
  plot <- 
    ggplot(data, aes(x = type_train, y = type_test, fill = m)) +
    geom_tile(color = 'white', lwd = 1.5, linetype = 1) +
    geom_tile(data = data %>% filter(diagonal == "Diagonal"),
              color = '#C66AC4', lwd = 3, linetype = 1) +
    geom_richtext(aes(label = round(m, 2)),
                  fill = 'white',
                  label.padding = unit(0.1, "lines"),
                  text.colour = "black",
                  fontface = "bold",
                  label.colour = "black") +
    scale_fill_viridis_c() +
    labs(y = "Test community", x = "Train community", fill = "Mean F1") +
    coord_fixed() 
  return(plot)
}

inter_community_f1_long <- 
  read_csv(paste0(path_intermediate_results, "metrics_type_f1score_df_long.csv")) %>% 
  filter(metric %in% c('f1')) %>%
  mutate(metric=factor(metric, levels=c('f1'), labels=c('F1'))) %>%
  mutate(type_train=gsub("\\['|'\\]", "", type_train)) %>% 
  mutate(type_test=gsub("\\['|'\\]", "", type_test)) %>% 
  mutate(type_train = case_when(type_train == "Plant-Seed Dispersers', 'Plant-Pollinator', 'Plant-Herbivore', 'Host-Parasite" ~ "All",
                                type_train == "Plant-Seed Dispersers', 'Plant-Herbivore', 'Host-Parasite" ~ "No PP",
                                type_train == 'Plant-Seed Dispersers' ~ 'PSD',
                                type_train == 'Plant-Pollinator' ~ 'PP',
                                type_train == 'Plant-Herbivore' ~ 'PH',
                                type_train == 'Host-Parasite' ~ 'HP')) %>% 
  mutate(type_test = case_when(type_test == "Plant-Seed Dispersers', 'Plant-Pollinator', 'Plant-Herbivore', 'Host-Parasite" ~ "All",
                               type_test == "Plant-Seed Dispersers', 'Plant-Herbivore', 'Host-Parasite" ~ "No PP",
                               type_test == 'Plant-Seed Dispersers' ~ 'PSD',
                               type_test == 'Plant-Pollinator' ~ 'PP',
                               type_test == 'Plant-Herbivore' ~ 'PH',
                               type_test == 'Host-Parasite' ~ 'HP')) %>% 
  mutate(type_train = factor(type_train, levels=c("All", "No PP", "HP", "PSD", "PH", "PP"))) %>% 
  mutate(type_test = factor(type_test, levels=rev(c("All", "No PP", "HP", "PSD", "PH", "PP"))))
  

## Plot for the paper ---------------------------------------------------------

plot_tibble <- 
  inter_community_f1_long %>%
  # filter(!type_train %in% c("No PP")) %>%
  # filter(!type_test %in% c("No PP")) %>%
  group_by(type_train, type_test, metric) %>%
  summarise(m = mean(value, na.rm = TRUE)) %>%
  mutate(diagonal = ifelse(type_train == type_test, "Diagonal", "Non-Diagonal"))

highlight_rect <- data.frame(
  xmin = as.numeric(as.factor(c("All"))),
  xmax = as.numeric(as.factor(c("No PP"))),
  ymin = as.numeric(as.factor(c("PP"))),
  ymax = as.numeric(as.factor(c("PSD")))
)


# Write figure source data according to NEE guidelines
write_csv(plot_tibble, '/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Fig_5_source_data.csv')

cross_community_prediction <- 
  plot_tibble %>% 
  create_heatmap_plot() +
  geom_rect(aes(xmin = 2.5, xmax = 6.5, ymin = 0.5, ymax = 4.5),
            color = 'blue', linetype = "dashed", fill = NA, linewidth = 1.5)+
  paper_figs_theme_main_NEE

cross_community_prediction
export_fig(cross_community_prediction, 'cross_community_prediction.pdf', 8, 8)


# Model bounds ----------------------------------------------------------------
bounds_summary_df <- read_csv(paste0(path_intermediate_results, "bounds_summary_df.csv"))
bounds_summary_df_transductive <- read_csv(paste0(path_intermediate_results, "bounds_summary_df_transductive.csv"))


merics_for_plot <- c("recall", "precision", "f1")

bounds_ILP_TLP <- 
  as_tibble(
    bind_rows(
      bounds_summary_df_transductive %>% filter(metric %in% merics_for_plot) %>% mutate(model='TLP'),
      bounds_summary_df %>% filter(metric %in% merics_for_plot) %>% mutate(model='ILP')
    )) %>% 
  # mutate((metric=factor(metric, levels=c("recall",'f1',"specificity",'mcc')))) %>% 
  mutate(across(metric, ~factor(., levels=merics_for_plot))) %>% 
  left_join(metrics, by = "metric")

bounds_ILP_TLP %>% 
  filter(fraction == 0.15) %>% 
  filter(metric == 'f1') %>% 
  select(fraction, metric, avg_lower_bound, avg_upper_bound, model)

# Write figure source data according to NEE guidelines
write_csv(bounds_ILP_TLP, '/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Extended_Fig_1_source_data.csv')


model_bounds_ILP_TLP <- 
  bounds_ILP_TLP %>% 
  ggplot(aes(x = fraction)) +
  geom_line(aes(y = avg_fixed_value, color = model), linetype = "dashed") +  # Plot average fixed value
  geom_line(aes(y = avg_lower_bound, color = model)) +  # Plot average lower bound
  geom_line(aes(y = avg_upper_bound, color = model)) +  # Plot average upper bound
  geom_ribbon(aes(ymin = ci_lower_lower_bound, ymax = ci_upper_lower_bound, fill = model), alpha=0.1) +  # Plot bounds as a ribbon with CI
  geom_ribbon(aes(ymin = ci_lower_upper_bound, ymax = ci_upper_upper_bound, fill = model), alpha=0.1) +  # Plot bounds as a ribbon with CI
  # geom_ribbon(aes(ymin = ci_lower_fixed, ymax = ci_upper_fixed), fill = "#f4a582", alpha = 0.5) +  # CI for fixed_value
  facet_wrap(~ metric_label) +  # Create a separate panel for each metric with custom labels
  scale_x_continuous(breaks = seq(0.05, 0.5, by = 0.1), limits = c(0.05, 0.5)) +  # Set x axis limits and breaks
  labs(x = 'Fraction of true missing links', y='Metric value', fill='Model', color='Model')+
  scale_color_manual(values = c('orange','#861ea5'), labels=c('ILP (ML)','TLP (ML)','SBM','MC','C')) +
  scale_fill_manual(values = c('orange','#861ea5'), labels=c('ILP (ML)','TLP (ML)','SBM','MC','C')) +
  geom_segment(data = . %>% filter(metric_label == "F1"),
               aes(x = 0.24, xend = 0.15, y = 0.742, yend = 0.742), 
               colour = "red", linewidth = 1, 
               arrow = arrow(length = unit(0.3, "cm")))+
  geom_segment(data = . %>% filter(metric_label == "F1"),
               aes(x = 0.24, xend = 0.15, y = 0.582, yend = 0.582), 
               colour = "red", linewidth = 1, 
               arrow = arrow(length = unit(0.3, "cm")))+
  paper_figs_theme_main_NEE + 
  theme(legend.position = c(0.9,0.82),
        strip.text = element_text(size = 14))

model_bounds_ILP_TLP

export_fig(model_bounds_ILP_TLP, 'extended_data_figure_1.pdf', 10, 5)

# Compare modeling approaches (e.g., RF vs XGboost) ----------------------------

# Load the dataset
metrics_multi_df_long <-
  read_csv(paste0(path_intermediate_results, "metrics_multi_df_long.csv")) %>% 
  left_join(metrics, by = "metric")

# Plot
SI_model_comparison <- 
  ggplot(metrics_multi_df_long, aes(x = metric_label, y = value, fill = model)) +
  geom_col(position = "dodge2", width = 0.7) +
  labs(y = "Value", fill = "Model") +
  paper_figs_theme_SI_NEE +
  theme(legend.position = "bottom",
        axis.title.x = element_blank(),
        plot.margin = unit(c(0.2,1,0.2,0.2), "cm"))

SI_model_comparison
export_fig(SI_model_comparison, 'SI_model_comparison.pdf', 10, 5)

# Compare sampling strategies --------------------------------------------------

# Random and weighted sampling (using only weighted dataset)
compare_models_noBias_metrics_df <-
  read_csv(paste0(path_intermediate_results, 'compare_models_noBias_metrics_df.csv')) %>% 
  mutate(sampling = 'Uniform')
compare_models_highDegBias_metrics_df <- 
  read.csv(paste0(path_intermediate_results, 'compare_models_highDegBias_metrics_df.csv')) %>% 
  mutate(sampling = 'Biased (high)')
compare_models_lowDegBias_metrics_df <- 
  read.csv(paste0(path_intermediate_results, 'compare_models_lowDegBias_metrics_df.csv')) %>% 
  mutate(sampling = 'Biased (low)')

compare_sampling <- 
  bind_rows(compare_models_noBias_metrics_df, compare_models_highDegBias_metrics_df, compare_models_lowDegBias_metrics_df) %>%
  left_join(metrics, by = "metric") %>% 
  left_join(models, by = "model")

SI_sampling_strategy <- 
compare_sampling %>% 
  ggplot(aes(x= metric_label, y = value, fill = sampling)) +
  geom_boxplot() +
  labs(y = "Value", x = "Metric", fill = "Sampling") +
  paper_figs_theme_SI_NEE +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


SI_sampling_strategy
export_fig(SI_sampling_strategy, 'SI_sampling_strategy.pdf', 10, 5)



# Sensitivity analysis for link removal ----------------------------------------

# Define the threshold for classification
threshold <- 0.5

link_removal_df <-
  read_csv(paste0(path_intermediate_results, 'sensitivity_results.csv')) %>% 
  mutate(p=1-frac_test) %>% 
  mutate(p=factor(p)) %>% 
  # Categorize each observation
  mutate(
    category = case_when(
      y_proba > threshold & y_true == 1 ~ "TP",
      y_proba > threshold & y_true == 0 ~ "FP",
      y_proba <= threshold & y_true == 1 ~ "FN",
      y_proba <= threshold & y_true == 0 ~ "TN",
      TRUE ~ NA_character_  # Handle unexpected cases
    )
  ) 

# Calculate performance metrics
link_removal_metrics <- 
  link_removal_df %>% 
  group_by(p,community,network) %>%
  summarize(
    TP = sum(category == "TP", na.rm = TRUE),
    FP = sum(category == "FP", na.rm = TRUE),
    TN = sum(category == "TN", na.rm = TRUE),
    FN = sum(category == "FN", na.rm = TRUE),
    .groups = "drop"  # Ungroup after summarizing
  ) %>%
  mutate(
    Precision = if_else((TP + FP) > 0, TP / (TP + FP), NA_real_),
    Recall = if_else((TP + FN) > 0, TP / (TP + FN), NA_real_),
    Specificity = if_else((TN + FP) > 0, TN / (TN + FP), NA_real_),
    BA = (Recall + Specificity) / 2,
    F1 = if_else((Precision + Recall) > 0, 2 * (Precision * Recall) / (Precision + Recall), NA_real_)
  ) 

# Reshape the metrics data to long format
SI_link_removal_sensitivity <- 
  link_removal_metrics %>%
  pivot_longer(
    cols = c(Precision, Recall, Specificity, BA, F1),
    names_to = "metric",
    values_to = "value"
  ) %>% 
  # Create boxplots of evaluation metrics
  ggplot(aes(x = p, y = value, fill=metric)) +
  geom_boxplot(outlier.alpha = 0.2, position = position_dodge(width = 0.8)) +
  facet_wrap(~metric, scales = "free_y") + 
  labs(
    x = "Proportion links removed",
    y = "Metric value (average across networks)",
    fill='Metric'
  ) +
  scale_fill_manual(values = c("#E06A5A", "#99008C", "#EACA00", "#1984c5", "#4DAF4A")) +
  paper_figs_theme_SI_NEE+
  theme(
    legend.position = 'none',
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )
SI_link_removal_sensitivity
export_fig(SI_link_removal_sensitivity, 'SI_link_removal_sensitivity.pdf', 10, 5)

# Evaluate prediction threshold ------------------------------------------------

## Precision-recall tradeoff -------------------------------------------------
# Set labels
new_names <- c(
  'All' = "A. All communities",
  "Host-Parasite" = "B. Host-Parasite",
  "Plant-Pollinator" = "C. Plant-Pollinator",
  "Plant-Seed Dispersers" = "D. Plant-Seed Dispersers",
  "Plant-Herbivore" = "E. Plant-Herbivore"
)

SI_pr_tradeoff <- 
  ggplot(pr_df, aes(x = threshold)) +
  geom_line(aes(y = precision, color = "Precision"), linetype = "solid") +
  geom_line(aes(y = recall, color = "Recall"), linetype = "solid") +
  geom_line(aes(y = f1, color = "F1 Score"), linetype = "solid") +
  scale_color_manual(values = c("Precision" = "#E06A5A", "Recall" = "#99008C", "F1 Score" = "#EACA00")) +
  labs(x = "Threshold", y = "Precision/Recall score", color = "Metric") +
  paper_figs_theme_SI_NEE +
  facet_wrap(~community, labeller = as_labeller(new_names)) +
  theme(axis.text.x=element_text(angle=-45, hjust=0))

# Display the plot
print(SI_pr_tradeoff)
export_fig(SI_pr_tradeoff, 'SI_pr_tradeoff.pdf', 10, 5)

## Distribution of predicted probabilities -------------------------------------

# Load the dataframe
test_data <- read_csv(paste0(path_intermediate_results, "test_data.csv"))

# Set labels
labels_mapping <- c(
  "Non-existing Links" = "A. Non-existing Links",
  "Subsampled Links" = "B. Subsampled Links",
  "Host-Parasite" = "A. Host-Parasite",
  "Plant-Pollinator" = "B. Plant-Pollinator",
  "Plant-Seed Dispersers" = "C. Plant-Seed Dispersers",
  "Plant-Herbivore" = "D. Plant-Herbivore"
)

# Plot the distribution with subplots
SI_probabilities <- 
  test_data %>%
  mutate(class = ifelse(y_true >= 0.5, "Subsampled Links", "Non-existing Links")) %>% # Create a new column to categorize the data based on the threshold
  ggplot(aes(x = y_proba)) +
  geom_histogram(binwidth = 0.05, alpha = 0.8, fill = "steelblue", color = "white") +
  geom_vline(aes(xintercept = 0.5), color = "red", linetype = "dashed") +
  scale_color_manual(name = "", values = c("Threshold" = "red"), labels = c("Threshold" = "threshold=0.5")) +	
  labs(x = "Probability", y = "Frequency", color = "legend") + #title = "Distribution of the predicted probabilities", 
  paper_figs_theme_SI_NEE +
  facet_wrap(~ class, nrow = 1, scales = "free_y", labeller = as_labeller(labels_mapping)) +
  theme(legend.position='none',
        plot.margin = unit(c(0.2,0.4,0.2,0.2), "cm"))

# Display the plot
print(SI_probabilities)
export_fig(SI_probabilities, 'SI_probabilities.pdf', 10, 5)

# How many links classified?
test_data %>%
  mutate(class = ifelse(y_true >= 0.5, "Subsampled Links", "Non-existing Links")) %>%
  group_by(class) %>%
  summarise(
    count_condition = sum(if_else(class == "Subsampled Links", y_proba > 0.5, y_proba < 0.5), na.rm = TRUE),
    total = n(),
    proportion = count_condition / total
  )

## Distribution (density) of link probabilities: per community ------------------

# Plot
SI_probabilities_community <- 
  test_data %>%
  mutate(class = ifelse(y_true >= 0.5, "Subsampled Links", "Non-existing Links")) %>%
  ggplot(aes(x = y_proba, fill = class)) +
  geom_density(alpha = 0.5) +
  # geom_histogram(binwidth = 0.05, color = "white", alpha = 0.5, position = 'identity') +
  geom_vline(aes(xintercept = 0.5), color = "red", linetype = "dashed") +
  labs(x = "Probability", y = "Density") +
  scale_fill_manual(values = c("Non-existing Links" = "steelblue", "Subsampled Links" = "orange"), labels = labels_mapping) +
  facet_wrap(~ community, labeller = as_labeller(labels_mapping)) +
  paper_figs_theme_SI_NEE +
  theme(legend.title = element_blank(),
        axis.text.x=element_text(angle=-45, hjust=0))

# Display the plot
print(SI_probabilities_community)
export_fig(SI_probabilities_community, 'SI_probabilities_community.pdf', 10, 5)


# Testing excluded networks ----------------------------------------------------

## Plot network properties -----------------------------------------------------

# Step 1: Read and Inspect Data
excluded_networks <- read_csv(paste0(path_raw_results, "filtered_networks.csv")) %>% 
  mutate(exclusion_criteria = ifelse(connectance_filtered, 'C', 'N'))  # Create a new column to indicate excluded networks

excluded_networks %>% 
  group_by(exclusion_criteria, community) %>%
  count()

# Step 2: Calculate Counts per Community Type
community_counts <- excluded_networks %>%
  group_by(community) %>%
  summarise(count = n()) %>%
  ungroup()

# Optional: Ensure 'community' is a factor with the same order as in the main plot
community_counts$community <- factor(community_counts$community, levels = unique(filtered_networks$community))

# Step 3: Create the Main Scatter Plot
main_plot <- excluded_networks %>% 
  ggplot(aes(x = net_size, y = connectance, color = community)) +
  geom_point(size=2) +
  geom_hline(yintercept = 0.1, linetype = "dashed") +
  geom_vline(xintercept = 25, linetype = "dashed") +
  scale_color_manual(values = setNames(communities$community_color_dark, communities$community)) +
  labs(x = "Network Size", y = "Connectance", color = "Community Type") +
  paper_figs_theme_SI_NEE +
  theme(
    legend.position = "none"  # Adjust legend position as needed
  )

# Step 4: Create the Inset Bar Plot
inset_plot <- community_counts %>%
  ggplot(aes(x = community, y = count, fill = community)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  scale_fill_manual(values = setNames(communities$community_color_dark, communities$community)) +
  labs(x = "Community Type", y = "Count") +
  paper_figs_theme_SI_NEE +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.title.x = element_blank()
  )

# Step 5: Combine Main Plot and Inset Plot
# Define the position and size of the inset plot
SI_excluded_networks_properties <- 
  ggdraw() +
  draw_plot(main_plot) +
  draw_plot(
    inset_plot,
    x = 0.35,  # x position (0 to 1)
    y = 0.4,  # y position (0 to 1)
    width = 0.5,  # width relative to main plot
    height = 0.5  # height relative to main plot
  )

# Step 6: Display the Combined Plot
print(SI_excluded_networks_properties)


## Compare evaluations with used networks --------------------------------------
# Define the threshold for classification
threshold <- 0.5

test_excluded_nets_df <-
  read_csv(paste0(path_intermediate_results, 'results_filtered_networks.csv')) %>% 
  select(-community) %>% 
  inner_join(excluded_networks, by = "network") %>%
  # Categorize each observation
  mutate(
    category = case_when(
      y_proba > threshold & y_true == 1 ~ "TP",
      y_proba > threshold & y_true == 0 ~ "FP",
      y_proba <= threshold & y_true == 1 ~ "FN",
      y_proba <= threshold & y_true == 0 ~ "TN",
      TRUE ~ NA_character_  # Handle unexpected cases
    )
  ) 

# Calculate evaluation metrics
excluded_nets_metrics <- 
  test_excluded_nets_df %>% 
  group_by(community,network,exclusion_criteria) %>%
  summarize(
    TP = sum(category == "TP", na.rm = TRUE),
    FP = sum(category == "FP", na.rm = TRUE),
    TN = sum(category == "TN", na.rm = TRUE),
    FN = sum(category == "FN", na.rm = TRUE),
    .groups = "drop"  # Ungroup after summarizing
  ) %>%
  mutate(
    Precision = if_else((TP + FP) > 0, TP / (TP + FP), NA_real_),
    Recall = if_else((TP + FN) > 0, TP / (TP + FN), NA_real_),
    Specificity = if_else((TN + FP) > 0, TN / (TN + FP), NA_real_),
    BA = (Recall + Specificity) / 2,
    F1 = if_else((Precision + Recall) > 0, 2 * (Precision * Recall) / (Precision + Recall), NA_real_)
  ) %>% 
  # Reshape the metrics data to long format
  pivot_longer(
    cols = c(Precision, Recall, Specificity, BA, F1),
    names_to = "metric",
    values_to = "value"
  )

# Take this from the ILP_vs_TLP section:
included_nets_metrics <- 
  compare_models_metrics_df %>% 
  mutate(metric = factor(metric, levels = metrics$metric, labels =  metrics$metric_label)) %>%
  # Filter to keep only the desired metrics
  filter(metric %in% desired_metrics) %>%
  # Reorder the metric factor according to the desired order
  mutate(metric = factor(metric, levels = desired_metrics)) %>%
  mutate(model = factor(model, levels = models$model, labels = models$label)) %>%
  filter(model == 'ILP')

SI_excluded_vs_included <- 
  bind_rows(
    excluded_nets_metrics %>% 
      filter(metric %in% c('F1','BA','Precision','Recall',"Specificity")) %>%
      mutate(type='Excluded') %>%
      select(type,exclusion_criteria,community,metric,value),
    included_nets_metrics %>% 
      filter(metric %in% c('F1','BA','Precision','Recall',"Specificity")) %>%
      mutate(type='Included') %>%
      mutate(exclusion_criteria = 'Included') %>% 
      select(type,exclusion_criteria,community,metric,value)
  ) %>%  
  ggplot(aes(x = metric, y = value, fill=type)) +
  geom_boxplot(outlier.alpha = 0.2, notch=T) +
  scale_fill_manual(values = c("#F7E0AE", "orange")) +
  labs(
    x = "Evaluation metrics",
    y = "Metric value",
    fill='Criteria'
  ) +
  paper_figs_theme_SI_NEE +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

SI_excluded_vs_included
# export_fig(SI_unsued_vs_used, 'SI_unsued_vs_used.pdf', 10, 5)

SI_excluded_networks <- 
cowplot::plot_grid(SI_excluded_networks_properties+
                     theme(plot.margin = margin(0,0,0,0)),
                   SI_excluded_vs_included+
                     theme(plot.margin = margin(0,0,0,0), 
                           legend.position = 'top',
                           legend.title = element_blank()),
                   nrow = 1, 
                   # align = "v",
                   # axis = "tb",
                   scale=c(0.88,1),
                   rel_widths = c(0.55,0.45),
                   labels = c("a", "b"))
                   
SI_excluded_networks
export_fig(SI_excluded_networks, 'SI_excluded_networks.pdf', 12, 8)



# Case study -------------------------------------------------------------------

## Prepare the data for the case study -----------------------------------------
# Load necessary libraries
# library(readxl)
# 
# # Load the data
# file_path <- "case_study/41559_2017_BFs415590170101_MOESM36_ESM.xlsx"
# data <- read_excel(file_path, sheet = "NATECOLEVOL-16040015-s03.csv")
# 
# # Separate data by year
# data_split <- data %>%
#   group_split(YearCollected)
# 
# # Create species interaction matrices for each year
# interaction_matrices <- map(data_split, function(df) {
#   df %>%
#     select(-YearCollected, -Host) %>%
#     pivot_longer(cols = everything(), names_to = "Species", values_to = "Value") %>%
#     pivot_wider(names_from = "Species", values_from = "Value", values_fill = 0) %>%
#     column_to_rownames("Host")
# })
# 
# 
# x %>% select(-YearCollected) %>%
#   # mutate(across(-Host, ~ ifelse(. > 0, 1, 0))) # Binarize values
#   write_csv(paste0(y,'.csv'))
# 
# interaction_matrices <- map(data_split, function(df) {
#   y=df$YearCollected[1]
#   
#   # Aggregate rows by 'Host', summing across columns
#   aggregated_matrix <- df %>%
#     select(-YearCollected) %>%
#     group_by(Host) %>%
#     summarise(across(everything(), \(x) sum(x, na.rm = TRUE))) # Sum across columns for duplicate species
#   
#   # Binarize the aggregated matrix
#   binary_matrix <- aggregated_matrix %>%
#     mutate(across(-Host, ~ ifelse(. > 0, 1, 0))) # Convert sums to binary (1/0)
#   
#   # Save the binary matrix to a CSV file
#   binary_matrix %>%
#     write_csv(paste0('case_study/',y, ".csv"))
#   
#   # aggregated_matrix %>%
#   #   write_csv(paste0(y, "_w.csv"))
# })

## Overall prediction evaluation------------------------------------------------

# Read the predicted interactions
# Define the threshold for classification
threshold <- 0.5

case_study_df <-
  read_csv(paste0(path_intermediate_results, 'case_study.csv')) %>% 
  # Categorize each observation
  mutate(
    category = case_when(
      class == -1 & y_pred == 1 ~ "TP",
      class == -1 & y_pred == 0 ~ "FN",
      class == 0 & y_pred == 1 ~ "FP",
      class == 0 & y_pred == 0 ~ "TN",
      TRUE ~ NA_character_  # Handle unexpected cases
    )
  ) 


## Relationship between node degree and prediction accuracy --------------------

### 1. Compute node degrees -----------------------------------------------------

# Host degree by year (how many unique parasite species each host truly interacted with)
host_deg <- case_study_df %>%
  filter(class %in% c(1, -1)) %>%   # these represent actual links
  group_by(year, lower_level) %>%
  summarize(host_degree = n_distinct(higher_level), .groups = "drop")

# Parasite degree by year (how many unique host species each parasite truly interacted with)
parasite_deg <- case_study_df %>%
  filter(class %in% c(1, -1)) %>%   # these represent actual links
  group_by(year, higher_level) %>%
  summarize(parasite_degree = n_distinct(lower_level), .groups = "drop")

# Join degrees back to the main data frame

case_study_df %<>%
  left_join(host_deg, by = c("year", "lower_level")) %>%
  left_join(parasite_deg, by = c("year", "higher_level"))


### 2. Host and parasite taxonomy ----------------------------------------------

# Define the species lists
rodent_species <- c(
  "Apodemus_agrarius", "Apodemus_peninsulae", "Arvicola_terrestris",
  "Cricetus_cricetus", "Eutamias_sibiricus", "Microtus_agrestis",
  "Microtus_arvalis", "Microtus_gregalis", "Microtus_minutus",
  "Microtus_oeconomus", "Mus_musculus", "Myodes_glareolus",
  "Myodes_rufocanus", "Myodes_rutilus", "Sicista_betulina"
)

eulipotyphla_species <- c(
  "Asioscalops_altaica",
  "Sorex_araneus", "Sorex_caecutiens", "Sorex_isodon",
  "Sorex_minutus", "Sorex_tundrensis", "Neomys_fodiens"
)

flea_species <- c(
  "Amphipsylla_sibirica", "Ceratophyllus_indages", "Ctenophthalmus_assimilis",
  "Megabothris_turbidus", "Megabothris_rectangulatus", "Amalaraeus_penicilliger",
  "Doratopsylla_birulai", "Frontopsylla_elata", "Histrichopsylla_talpae",
  "Leptopsylla_segnis", "Neopsylla_acanthina", "Neopsylla_mana",
  "Palaeopsylla_soricis", "Rhadinopsylla_integella"
)


# Tag each host as Rodentia or Eulipotyphla
case_study_df  %<>%
  mutate(
    host_order = case_when(
      lower_level %in% rodent_species ~ "Rodentia",
      lower_level %in% eulipotyphla_species ~ "Eulipotyphla",
      TRUE ~ NA_character_
    )
  )

# Classify parasites as 'Flea' vs. 'Mite'
case_study_df  %<>%
  mutate(parasite_class = case_when(
    higher_level %in% flea_species ~ "Flea",
    TRUE ~ "Mite"  # everything else
  ))



### 3. Restrict to test set & compute correctness ------------------------------

case_study_test <-  
  case_study_df %>%
  filter(class %in% c(0, -1))

### 4a. Prediction per year ------------------------------------------



# Write figure source data according to NEE guidelines
case_study_test %>%
  group_by(year) %>%
  calc_conf_metrics() %>% 
  pivot_longer(
    cols = c(accuracy, precision, recall, balanced_accuracy, f1), 
    names_to = "metric", 
    values_to = "value"
  ) %>%
  write_csv(paste0(path_intermediate_results, 'case_study_metrics.csv'))ic != "accuracy") %>% 
  write_csv( '/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Fig_6a_source_data.csv')


case_study_yearly_evaluation <- 
  case_study_test %>%
  group_by(year) %>%
  calc_conf_metrics() %>% 
  pivot_longer(
    cols = c(accuracy, precision, recall, balanced_accuracy, f1), 
    names_to = "metric", 
    values_to = "value"
  ) %>%
  filter(metric != "accuracy") %>%
  ggplot(aes(x = year, y = value, color = metric)) +
  geom_point() +
  geom_line() +
  labs(
    x = "Year",
    y = "Value",
    color = "Metric"
  ) +
  # rename the color legend entries
  scale_color_manual(
    labels = c(
      "precision"         = "Precision",
      "recall"            = "Recall",
      "balanced_accuracy" = "Balanced accuracy",
      "f1"                = "F1"
    ),
    values = c("#E06A5A", "#99008C", "#EACA00", "#1984c5")
  ) +
  paper_figs_theme_main_NEE+
  theme(legend.position = c(0.3,0.5))


case_study_yearly_evaluation
# export_fig(case_study_yearly_evaluation, 'case_study_yearly_evaluation.pdf', 10, 5)

SI_case_study_parasites_vs_hosts_per_year <- 
bind_rows(
  host_degree_evaluation <- 
    case_study_test %>%
    group_by(year,host_degree) %>%
    calc_conf_metrics() %>% 
    rename(k=host_degree) %>%
    mutate(group='Hosts'),
  
  parasite_degree_evaluation <- 
    case_study_test %>%
    group_by(year,parasite_degree) %>%
    calc_conf_metrics() %>% 
    rename(k=parasite_degree) %>%
    mutate(group='Parasites')
) %>% 
  ggplot(aes(x = k, y = f1, color = group)) +
  geom_point() +
  facet_wrap(~year)+
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Degree",
    y = "F1",
    color = "Group"
  ) +
  paper_figs_theme_SI_NEE

SI_case_study_parasites_vs_hosts_per_year
export_fig(SI_case_study_parasites_vs_hosts_per_year, 'SI_case_study_parasites_vs_hosts_per_year.pdf', 10, 5)

### 4b. Prediction across all years ------------------------------------------

# Write figure source data according to NEE guidelines
bind_rows(
  host_degree_evaluation <- 
    case_study_test %>%
    group_by(host_degree) %>%
    calc_conf_metrics() %>% 
    rename(k=host_degree) %>%
    mutate(group='Hosts'),
  
  parasite_degree_evaluation <- 
    case_study_test %>%
    group_by(parasite_degree) %>%
    calc_conf_metrics() %>% 
    rename(k=parasite_degree) %>%
    mutate(group='Parasites')
) %>% 
write_csv( '/Users/shai/Dropbox (Personal)/Apps/Overleaf/2025 NEE/revision_2/NEE/Fig_6b_source_data.csv')


case_study_parasites_vs_hosts <- 
  bind_rows(
    host_degree_evaluation <- 
      case_study_test %>%
      group_by(host_degree) %>%
      calc_conf_metrics() %>% 
      rename(k=host_degree) %>%
      mutate(group='Hosts'),
    
    parasite_degree_evaluation <- 
      case_study_test %>%
      group_by(parasite_degree) %>%
      calc_conf_metrics() %>% 
      rename(k=parasite_degree) %>%
      mutate(group='Parasites')
  ) %>% 
  ggplot(aes(x = k, y = f1, color = group)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(
    x = "Degree",
    y = "F1",
    color = "Group"
  ) +
  paper_figs_theme_main_NEE+
  theme(legend.position = c(0.8,0.15))

case_study_parasites_vs_hosts
# export_fig(case_study_parasites_vs_hosts, 'case_study_parasites_vs_hosts.pdf', 10, 5)

case_study <- 
  cowplot::plot_grid(case_study_yearly_evaluation, 
                     case_study_parasites_vs_hosts, 
                     nrow = 1,
                     labels = c("a", "b"))
case_study
export_fig(SI_case_study, 'case_study.pdf', 10, 5)


### 4c. Prediction by taxonomy ------------------------------------------

# Host degree by taxonomy
host_taxonomay_evaluation <- 
  case_study_test %>%
  group_by(host_order, host_degree) %>%
  calc_conf_metrics() %>% 
  ggplot(aes(x = host_degree, y = f1, color = host_order)) +
  geom_point() +
  geom_smooth(method='lm') +
  # facet_wrap(~ year, scales = "free_x") +
  scale_color_manual(values = c("Rodentia" = "#03396c", "Eulipotyphla" = "#6497b1")) +
  labs(
    x = "Host Degree",
    y = "F1",
    color = "Host order"
  ) +
  xlim(c(0,40))+
  ylim(c(0,1))+
  paper_figs_theme_SI_NEE +
  theme(legend.position = 'inside',
        legend.position.inside = c(0.25,0.85))

# Host degree by taxonomy
parasite_taxonomay_evaluation <- 
  case_study_test %>%
  group_by(parasite_class, parasite_degree) %>%
  calc_conf_metrics() %>% 
  ggplot(aes(x = parasite_degree, y = f1, color = parasite_class)) +
  geom_point() +
  geom_smooth(method='lm') +
  # facet_wrap(~ year, scales = "free_x") +
  scale_color_manual(values = c("Flea" = "#ff084a", "Mite" = "#800000")) +
  labs(
    x = "Parasite degree",
    y = "F1",
    color = "Parasite class"
  ) +
  xlim(c(0,40))+
  ylim(c(0,1))+
  paper_figs_theme_SI_NEE+
  theme(legend.position = 'inside',
        legend.position.inside = c(0.25,0.85))


SI_case_study_taxonomy <- 
  cowplot::plot_grid(host_taxonomay_evaluation, parasite_taxonomay_evaluation, nrow = 1)
SI_case_study_taxonomy
export_fig(SI_case_study_taxonomy, 'SI_case_study_taxonomy.pdf', 10, 5)
