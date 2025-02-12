suppressPackageStartupMessages({

    library(tidyverse)
    library(igraph)
    library(bmotif)
    library(bipartite)
})

# Main function
topoFeatures <- function(edgelist){

  # Remove '0' links
  edgelist_compact = edgelist[edgelist$weight != 0,] # !! careful, removes singletons !!
    
  # Ignore weights
  edgelist$weight[edgelist$weight > 0] <- 1 
  
  # Convert edgelist to matrix
  matrix = bipartite::frame2webs(edgelist, varname = c("lower_level", "higher_level", "subsample_ID", "weight"), emptylist=FALSE)[[1]]
  
  #--------------------------------------------------------------------------------------------
    
  # Get network properties
  net_features = get_net_features(matrix)
  
  # Combine 'net_features' with 'edgelist'
  duprows <- colnames(edgelist) %in% colnames(net_features) # Take only the new values
  edgelist = cbind(edgelist[,!duprows], net_features)
  
  #--------------------------------------------------------------------------------------------
  
  # Get node properties
  nodes_features <- get_node_features(matrix)
  
  # Combine 'nodes_features' with 'edgelist', keep the same order of columns
  duprows <- colnames(edgelist) %in% colnames(nodes_features[[1]]) # Take only the new values
  nodes_features[[1]][setdiff(unique(edgelist$higher_level), rownames(nodes_features[[1]])),] <- 0 # Fill 0 for singletons 
  edgelist = merge(edgelist[,!duprows], nodes_features[[1]], by.x = "higher_level", by.y = 0, all.x = TRUE, sort = FALSE)[, union(names(edgelist[,!duprows]), names(nodes_features[[1]]))]
  
  duprows <- colnames(edgelist) %in% colnames(nodes_features[[2]]) # Take only the new values
  nodes_features[[2]][setdiff(unique(edgelist$lower_level), rownames(nodes_features[[2]])),] <- 0 # Fill 0 for singletons 
  edgelist = merge(edgelist[,!duprows], nodes_features[[2]], by.x = "lower_level", by.y = 0, all.x = TRUE, sort = FALSE)[, union(names(edgelist[,!duprows]), names(nodes_features[[2]]))]
  
  #--------------------------------------------------------------------------------------------
  
  # stolen code - more net features
  results = analyze_one_file(edgelist_compact[, c('lower_level', 'higher_level', 'weight')], 100)

  # Combine with 'edgelist'
  duprows <- colnames(edgelist) %in% colnames(results) # Take only the new values
  edgelist = cbind(edgelist[,!duprows], results)
  
  #--------------------------------------------------------------------------------------------
  
  return (edgelist)
  
}

# Network features
get_net_features <- function(matrix){
  
  net_features <- as.data.frame((networklevel(matrix,
                                           index = c(
                                             'connectance',
                                             'web asymmetry',
                                             'links per species',
                                             'number of compartments',
                                             'compartment diversity',
                                             #'cluster coefficient',
                                             #'degree distribution', #doing it separately, as it interfere with the dataframe creation
                                             #'mean number of shared partners',
                                             #'mean number of links',
                                            #  'togetherness',
                                            #  'C score',
                                             'V ratio',
                                            #  'discrepancy',
                                             'nestedness',
                                             'NODF',
                                             #'weighted nestedness', #weighted
                                             #'weighted NODF', #weighted
                                             #'ISA', #interaction strength asymmetry
                                             #'SA',
                                            #  'extinction slope',
                                             'robustness',
                                             'niche overlap',
                                             #'weighted cluster coefficient', #weighted
                                             #'modularity',
                                             #'partner diversity', #weighted
                                            #  'generality',
                                             'vulnerability',
                                             #'linkage density', #weighted
                                             #'weighted connectance', #weighted
                                             'Fisher alpha',
                                             'interaction evenness',
                                             #'Alatalo interaction evenness',
                                             'Shannon diversity',
                                            #  'functional complementarity',
                                             'H2'
                                             
                                           ), weighted=FALSE, fcweighted = FALSE, fcdist="euclidean")
  ))
  net_features=as.data.frame((net_features))
  colnames(net_features) <- c("value")

  # Motifs count
  motif_count = mcount(M = matrix, six_node = F, normalisation = F, mean_weight = F, standard_dev = F)[c("motif", "frequency")] # get motif count
  colnames(motif_count)[colnames(motif_count) == "frequency"] = "value" # rename column
  motif_count$motif <- paste("motif_", motif_count$motif, sep = "") # add prefix to motif names
  rownames(motif_count) <- motif_count$motif # set the 'motif' column as row names
  motif_count$motif <- NULL # remove 'motif' column

  # Motifs count - normalised
  motif_count_norm = mcount(M = matrix, six_node = F, normalisation = T, mean_weight = F, standard_dev = F)[c("motif", "normalise_sum")] # get motif count
  colnames(motif_count_norm)[colnames(motif_count_norm) == "normalise_sum"] = "value" # rename column
  motif_count_norm$motif <- paste("motif_", motif_count_norm$motif, "_norm", sep = "") # add prefix to motif names
  rownames(motif_count_norm) <- motif_count_norm$motif # set the 'motif' column as row names
  motif_count_norm$motif <- NULL # remove 'motif' column

  # Combine all features
  net_features = rbind(net_features, motif_count)
  net_features = rbind(net_features, motif_count_norm)

  return(t(net_features))
}

#--------------------------------------------------------------------------------------------
# Node features

get_node_features <- function(matrix){
  
  nodes_features <- specieslevel(matrix,
                              index = c(# "degree",
                                        # "normalised degree",
                                        "species strength", 
                                        "nestedrank", 
                                        "interaction push pull", 
                                        "PDI", 
                                        # "resource range", 
                                        "species specificity", 
                                        # "PSI", 
                                        "NSI", 
                                        "betweenness", 
                                        "closeness", 
                                        "Fisher alpha", 
                                        # "partner diversity", 
                                        # "effective partners",
                                        "d"
                                        # "dependence", 
                                        # "proportional generality", 
                                        # "proportional similarity"
                                        )
                              )
    
  nodes_features.HL = nodes_features[["higher level"]]
  nodes_features.LL = nodes_features[["lower level"]]
  
  drops <- c("weighted.betweenness","weighted.closeness")
  nodes_features.HL <- nodes_features.HL[ , !(names(nodes_features.HL) %in% drops)]
  nodes_features.LL <- nodes_features.LL[ , !(names(nodes_features.LL) %in% drops)]
  
  colnames(nodes_features.HL) <- paste(colnames(nodes_features.HL), "HL", sep = "_")
  colnames(nodes_features.LL) <- paste(colnames(nodes_features.LL), "LL", sep = "_")
  
  return(list(nodes_features.HL, nodes_features.LL))
}

#--------------------------------------------------------------------------------------------
# stolen code

analyze_one_file <- function(edge_list, MAX_SIZE) {
  
  g <- graph_from_data_frame(edge_list, directed=FALSE)
  V(g)$type <- V(g)$name %in% edge_list$higher_level

  ## centralities
  cent_between <- centr_betw(g, directed=FALSE)$centralization
  cent_close <- centr_clo(g, mode="total")$centralization
  cent_eigen <- centr_eigen(g)$centralization
  
  ## add to results
  return (tibble(
    # l1, l2, l3,
    # ipr1, ipr2, ipr3,
    cent_between, cent_close, cent_eigen
  ))
}