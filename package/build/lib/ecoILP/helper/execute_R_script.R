# run_topoFeatures.R
source("helper/topology_functions.R")

args <- commandArgs(trailingOnly = TRUE)
input_file <- args[1]
output_file <- args[2]

network <- read.csv(input_file, stringsAsFactors = FALSE)

features <- topoFeatures(network)

# drop columns which are not new features
drop_columns <- colnames(network) 
drop_columns <- drop_columns[drop_columns!='link_ID']
features = features[,!(names(features) %in% drop_columns)]

# Assuming you handle the column dropping in the R script
write.csv(features, output_file, row.names = FALSE)