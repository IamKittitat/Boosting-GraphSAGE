# Boosting-GraphSAGE
Replicating of [Automatic disease prediction from human gut metagenomic data using boosting GraphSAGE](https://www.researchgate.net/publication/369688964_Automatic_disease_prediction_from_human_gut_metagenomic_data_using_boosting_GraphSAGE)

## Steps
1. Calculate distance matrix (N,N) from OUT Table
   - Choose between Cosine Dissimilarity, Manhattan Dissimilarity
2. Distance Threshold Computation: To determine if 2 nodes is neighbor
   - Median value of the distance matrix
3. Calculate Neighbor threshold (max threshold each node can have)
   - Cal tau_healthy, tau_sick for imbalance data
4. Create Metagenomic Disease Graph (MD-Graph) using the distance matrix
   - Samples == Node | Edges are constructed based on the similarity of features
     1. Edge Construction: A_ij = 1, if (d_ij < t) and (tau_i <= tau)
     2. Edge Refinement: Handle sparse node
     3. Add features vectors to MD-graph nodes
5. Create Disease Prediction Network (DP-Net) + Ensemble GraphSAGE
   1. GraphSAGE
   2. Boosting: GraphSAGE + AdaBoost
      1. Initialization: Split train:val:test -> init weight
      2. Iterative: Ada

## Experimental Setup
- Comparison: ML, Ensemble ML, GNN, Ensemble GNN (Also include bagging GraphSAGE)
- Parameter Settings (Use 1 GraphSAGE Model)
  - Optimum no. of neighbor, best dissimilarity measure -> F1
  - Find no. of layers, sample number -> F1
    - * Learning rate, optimizer, hidden units, act func, loss func is fixed. (Ref 12,18 in table3)
  - Find no. of base classifiers, f
- Evaluation Metrics: F1, AUC, AUPRC
