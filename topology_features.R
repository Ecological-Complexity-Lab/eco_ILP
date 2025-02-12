suppressPackageStartupMessages({
  # library(yaml)
  library(foreach)
  library(doParallel)
  library(reshape)
  library(magrittr)
  library(tidyverse)
  library(tools)
  library(stringr)
  library(glue)
  library(assertthat)
  library(doSNOW) # parallel + progress bar fix
  
  library(igraph)
  library(bmotif)
  library(bipartite)
})

#devtools::install_github("comeetie/greed")
#library(greed)

machine = Sys.info()["sysname"] # get type of OS

if(machine == "Linux"){ # HPC
  dir_path = "/gpfs0/shai/users/barryb/link-predict/"
  cores = as.numeric(Sys.getenv('NSLOTS'))
  
} else { # Windows / MacOS
  dir_path = dirname(rstudioapi::callFun("getActiveDocumentContext")$path) # Rstudio
  cores = 2
}

setwd(dir_path) # Rstudio
message(paste('Number of cores: ', cores, '\n'))

parallel_processing = TRUE

# Import functions
source("helper/topology_functions.R")

# Arguments
args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

# Load data
message('Loading dataframe')
data <- read.csv(input_file, header=TRUE, stringsAsFactors=FALSE, check.names=FALSE, na.strings="") # | debug: , nrows = 10124
message('Done\n')

# drop (later) all columns which are not 'link_ID'
drop_columns <- colnames(data) 
drop_columns <- drop_columns[drop_columns!='link_ID']

# Get subsamples ids
ids = unique(data$subsample_ID)
len = length(ids)

message('Extracting topological features\n')
if (parallel_processing == TRUE){
  
  cl <- makeCluster(cores, outfile="/dev/null") # outfile="/dev/null": surpress the "Type: EXEC" and "Type: DONE" output on HPC
  registerDoSNOW(cl)
  
  pb <- txtProgressBar(min = 1, max = len, style = 3)
  
  progress <- function(i) setTxtProgressBar(pb, i)
  opts <- list(progress = progress)
  
  df <- foreach (i=1:len,
                 .combine=rbind,
                 .options.snow = opts,
                 .packages = c("magrittr", "purrr", "bipartite", "tidyverse", "igraph", "bmotif")) %dopar% {
                   id = ids[[i]]
                   edgelist = topoFeatures(data[data$subsample_ID == id, ])
                   return(edgelist)
                 }
  close(pb)
  stopCluster(cl)
  
} else { #no parallel | not efficiently implemented!! change that
  # Split to lists of subsamples
  message('Spliting datafame to list of subsamples edgelists\n')
  # data_lists = split(data, data$subsample_ID) 
  # edgelists <- lapply(data_lists, topoFeatures) #data_lists[1:8]
  # df = do.call("rbind", edgelists)
}

# drop columns which are not new features
df = df[,!(names(df) %in% drop_columns)]

# Update the new entries
cat('Exporting new dataframe\n')
write.csv(df, output_file, row.names = FALSE)
cat('Done')

