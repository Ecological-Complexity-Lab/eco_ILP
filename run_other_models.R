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

# Paths of data
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

## Set Variables

cores = as.numeric(Sys.getenv('NSLOTS'))
if (is.na(cores)){
  cores = 2
}
parallel_processing = T


## Load data

# Load subsamples (edgelists) data
message('Loading data\n')
subsamples_edge_lists <- read.csv(input_file)

# Define the test set
test_data <- subsamples_edge_lists

# scaling the means of the probabilities, excluding ecisting links
scale_probs <- function(network_list, probs_matrix){

    positions_to_modify <- which(network_list$obs != 1, arr.ind = TRUE)
    mean = mean(positions_to_modify)
    probs_matrix[positions_to_modify] <- probs_matrix[positions_to_modify] / mean

    return(probs_matrix)
}

# Main function - fit models
fit_models <- function(edgelist, models = c("SBM", "C", "MC", "CD", "SBM_C_avg", "MC_C_avg"), n=10) {
    
    # Convert edgelist to matrix
    matrix = bipartite::frame2webs(edgelist, varname = c("lower_level", "higher_level", "subsample_ID", "weight"), emptylist=FALSE)[[1]]
    
    # Create list object
    network_list <- CreateListObject(matrix)

    # Columns to bind
    cols = c()
    
    # Fit SBM
    if ("SBM" %in% models){
        SBM_ProbsMatrix <- FitSBM(network_list, n_SBM = n, G = NULL)$SBM_ProbsMat
        # SBM_ProbsMatrix = scale_probs(network_list, SBM_ProbsMatrix)
        SBM_df <- melt(SBM_ProbsMatrix) %>%
            rename('SBM_Prob' = value) %>%
            select(SBM_Prob)
        cols = c(cols, SBM_df)
    }

    # Fit Matching Centrality
    if ("MC" %in% models){
        MC_ProbsMatrix <- FitBothMandC(network_list, N_runs = n, maxit = 10000, method = "Nelder-Mead")$B_ProbsMat
        # MC_ProbsMatrix = scale_probs(network_list, MC_ProbsMatrix)
        MC_df <- melt(MC_ProbsMatrix) %>%
            rename('MC_Prob' = value) %>%
            select(MC_Prob)
        cols = c(cols, MC_df)
    }

    # Fit Centrality
    if ("C" %in% models){
        C_ProbsMatrix <- FitCentrality(network_list, N_runs = n, maxit = 10000, method = "Nelder-Mead")$C_ProbsMatrix
        C_df <- melt(C_ProbsMatrix) %>%
            rename('C_Prob' = value) %>%
            select(C_Prob)
        cols = c(cols, C_df)
    }

    # Fit Coverage Deficit
    if ("CD" %in% models){
        CD_ProbsMatrix <- CalcHostLevelCoverage(network_list)$C_defmatrix
        CD_ProbsMatrix[is.infinite(CD_ProbsMatrix)] <- 0 # fix inf values
        CD_ProbsMatrix = scale_probs(network_list, CD_ProbsMatrix)
        CD_df <- melt(CD_ProbsMatrix) %>%
            rename('CD_Prob' = value) %>%
            select(CD_Prob)
        cols = c(cols, CD_df)
    }

    # SBM & Coverage (averaging)
    if ("SBM_C_avg" %in% models){
        SBM_C_avg = SBM_ProbsMatrix + CD_ProbsMatrix
        SBM_C_avg_df <- melt(SBM_C_avg) %>%
            rename('SBM_C_avg' = value) %>%
            select(SBM_C_avg)
        cols = c(cols, SBM_C_avg_df)
    }

    # Matching Centrality & Coverage (averaging)
    if ("MC_C_avg" %in% models){
        MC_C_avg = MC_ProbsMatrix + CD_ProbsMatrix
        MC_C_avg_df <- melt(MC_C_avg) %>%
            rename('MC_C_avg' = value) %>%
            select(MC_C_avg)
        cols = c(cols, MC_C_avg_df)
    }

    # -----
    # Testing
    # MC_ProbsMatrix[which(network_list$obs == 1, arr.ind = TRUE)]
    # MC_ProbsMatrix[which(network_list$obs != 1, arr.ind = TRUE)]
    # MC_ProbsMatrix_no_TP = MC_ProbsMatrix[which(network_list$obs != 1, arr.ind = TRUE)]
    # MC_ProbsMatrix_no_TP/mean(MC_ProbsMatrix_no_TP)
    # -----

    # Create dataframe
    result <- expand.grid(network_list$HostNames, network_list$WaspNames, stringsAsFactors = FALSE) %>%
        rename('lower_level' = Var1, 'higher_level' = Var2) %>%
        bind_cols(cols)
    
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
            fit_models(as.data.frame(.x), models = c("SBM", "C", "MC"))
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
write.csv(pred_df %>% select(link_ID, SBM_Prob, C_Prob, MC_Prob), output_file, row.names = FALSE)

cat('\nDone')