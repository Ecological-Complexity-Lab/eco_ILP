## Import libraries
suppressPackageStartupMessages({
  library(dplyr)
    library(reshape2)
    library(purrr)
    library(tidyr)
    library(cassandRa)

    library(progress)
    library(foreach)
    library(doSNOW) # parallel + progress bar fix
})

## Set important paths

# Path of the main folder
path_main_dir = "/gpfs0/shai/users/barryb/link-predict/" # HPC
# dir_path = dirname(rstudioapi::callFun("getActiveDocumentContext")$path) # Rstudio

# Paths of data by specific purpose
path_metadata = paste0(path_main_dir, "data/processed/") # Path of proccessed networks & features data, serving as complementary metadata
path_raw_results = paste0(path_main_dir, "results/raw/") # Path of raw results

## Set Variables

cores = as.numeric(Sys.getenv('NSLOTS'))
if (is.na(cores)){
  cores = 2
}
parallel_processing = T


## Load data

# Load subsamples (edgelists) data
message('Loading data\n')
subsamples_edge_lists <- read.csv(paste0(path_metadata, "networks/subsamples_edge_lists.csv"))

# Define the test set
test_data <- subsamples_edge_lists


# Main function - fit models
fit_models <- function(edgelist, n=10) {
    
    # Convert edgelist to matrix
    matrix = bipartite::frame2webs(edgelist, varname = c("lower_level", "higher_level", "subsample_ID", "weight"), emptylist=FALSE)[[1]]
    
    # Create list object
    network_list <- CreateListObject(matrix)
    
    # Fit SBM
    SBM_ProbsMat <-
        melt(FitSBM(network_list, n_SBM = n, G = NULL)$SBM_ProbsMat) %>%
        rename('SBM_Prob' = value) %>%
        select(SBM_Prob)

    # Fit centrality
    C_ProbsMatrix <-
        melt(FitCentrality(network_list, N_runs = n, maxit = 10000, method = "Nelder-Mead")$C_ProbsMatrix) %>%
        rename('C_Prob' = value) %>%
        select(C_Prob)

    # Create dataframe
    result <- expand.grid(network_list$HostNames, network_list$WaspNames, stringsAsFactors = FALSE) %>%
        rename('lower_level' = Var1, 'higher_level' = Var2) %>%
        bind_cols(SBM_ProbsMat, C_ProbsMatrix)
    
    return(result)

}


# Get subsamples ids
ids = unique(test_data$subsample_ID)#[1:3]
len = length(ids)

message('Fitting predictive models\n')
if (parallel_processing == TRUE){
  
  cl <- makeCluster(cores, outfile="/dev/null") # outfile="/dev/null": surpress the "Type: EXEC" and "Type: DONE" output on HPC
  registerDoSNOW(cl)
  
  pb <- txtProgressBar(min = 1, max = len, style = 3)

  progress <- function(i) setTxtProgressBar(pb, i)
  opts <- list(progress = progress)
  
  pred_df <- foreach (i=1:len,
                 .combine=rbind,
                 .options.snow = opts,
                 .packages = c("cassandRa", "bipartite", "dplyr", "reshape2")) %dopar% {
                   id = ids[[i]]
                   edgelist = fit_models(test_data[test_data$subsample_ID == id, ] %>% select(lower_level, higher_level, subsample_ID, weight))
                   edgelist$subsample_ID = id
                   return(edgelist)
                 }
  close(pb)
  stopCluster(cl)
  
} else { # not parallel processing
    
    # Set up a manual progress bar
    pb <- progress_bar$new(
    format = "Calculating [:bar] :percent",
    total = len, clear = FALSE, width = 60
    )

    # Apply fit_models function on each subsample
    pred_df <- test_data %>%
        filter(subsample_ID %in% ids) %>% # Filter only the subsamples in the test set
        select(lower_level, higher_level, subsample_ID, weight) %>%
        mutate(subsample_ID_copy = subsample_ID) %>% # Create a copy of subsample_ID
        group_by(subsample_ID_copy) %>%
        nest() %>%
        mutate(model = map(data, ~ {
            pb$tick() # Update the progress bar at each iteration
            fit_models(as.data.frame(.x))
        })) %>%
        unnest(cols = c(model)) %>%
        rename('subsample_ID' = subsample_ID_copy) %>%
        select(-data)

    # pb$close()
}

# restore link_ID column
pred_df <- merge(pred_df, test_data %>% select(link_ID, lower_level, higher_level, subsample_ID), by = c("subsample_ID", "lower_level", "higher_level"))

# Save the results
cat('Exporting new dataframe\n')
write.csv(pred_df %>% select(link_ID, SBM_Prob, C_Prob), paste0(path_raw_results, "other_models.csv"), row.names = FALSE)

cat('\nDone')



# Debugging
edgelist = test_data %>% filter(subsample_ID == 3) %>% select( lower_level, higher_level, subsample_ID, weight)
res = fit_models(edgelist)
