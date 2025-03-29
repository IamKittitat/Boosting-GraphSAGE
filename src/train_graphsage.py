import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import dense_to_sparse
from src.graphsage.model import GraphSAGE
from torch_geometric.data import Data

def train_graphsage(features_train, features_val, adj_matrix_train, labels_train, labels_val, edge_index_train, neighbor_val, embed_dim, lr, num_epochs, num_layers):    
    num_feats = features_train.shape[1]

    graphsage = GraphSAGE(num_feats, embed_dim, 2, num_layers=num_layers)
    optimizer = torch.optim.Adam(graphsage.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        graphsage.train()
        optimizer.zero_grad()
        out_train = graphsage(features_train, edge_index_train)
        loss = criterion(out_train, labels_train)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            val_prob = []
            for features_node, neighbor_node in zip(features_val, neighbor_val):
                features_neighbor = features_train[neighbor_node]
                features_subgraph = torch.cat([features_node.unsqueeze(0), features_neighbor], dim=0)
                num_neighbors = len(neighbor_node)
                new_edges = []
                for i in range(1, num_neighbors + 1):
                    new_edges.append([0, i])
                    new_edges.append([i, 0])
                for i in range(num_neighbors):  
                    for j in range(i+1, num_neighbors):
                        if adj_matrix_train[neighbor_node[i]][neighbor_node[j]] == 1:
                            new_edges.append([i+1, j+1])
                            new_edges.append([j+1, i+1])
                graphsage.eval()
                with torch.no_grad():
                    edge_index_new = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
                    new_data = Data(x=features_subgraph, edge_index=edge_index_new)
                    out_val = graphsage(new_data.x, new_data.edge_index)
                    probs = torch.softmax(out_val, dim=1)[:, 1][0].item()
                    val_prob.append(probs)
            
            val_prob = torch.tensor(val_prob, dtype=torch.float32)
            train_auc = roc_auc_score(labels_train.cpu().numpy(), torch.softmax(out_train, dim=1)[:, 1].cpu().detach().numpy())
            val_auc = roc_auc_score(labels_val.cpu().numpy(), val_prob.cpu().numpy())
            print(f"Epoch: {epoch:03d}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")

    return val_auc

def train_boosting_graphsage(features, adj_matrix, labels, embed_dim, lr, num_epochs, base_estimators, num_layers):
    def boosting_predict(val_idx, base_models, model_weights, edge_index):
        final_predictions = torch.zeros((len(val_idx), 2))
        for model, weight in zip(base_models, model_weights):
            out_val = model(features, edge_index)
            weight = torch.tensor(weight, dtype=torch.float32)
            probs = torch.softmax(out_val[val_idx], dim=1)
            final_predictions += weight * probs
        return final_predictions[:, 1]
    
    M = base_estimators
    base_models = []
    model_weights = []
    auc_scores = []
    num_feats = features.shape[1]
    edge_index, _ = dense_to_sparse(adj_matrix)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        print(f"Fold {fold + 1}/10")
        N = len(train_idx)
        weights = np.ones(N) / N
        
        for m in range(M):
            print(f"Base estimators: {m+1}/{M}")
            bootstrap_idx = np.random.choice(train_idx, size=N, replace=True, p=weights)
            graphsage = GraphSAGE(num_feats, embed_dim, 2, num_layers=num_layers)
            optimizer = torch.optim.Adam(graphsage.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            for _ in range(num_epochs):
                graphsage.train()
                optimizer.zero_grad()
                out = graphsage(features, edge_index)
                loss = criterion(out[bootstrap_idx], labels[bootstrap_idx])
                loss.backward()
                optimizer.step()

            graphsage.eval()
            with torch.no_grad():
                out = graphsage(features, edge_index)
                _, pred = out.max(dim=1)

            incorrect = np.array(pred[bootstrap_idx] != labels[bootstrap_idx])
            error_m = np.sum(weights * incorrect) / np.sum(weights)
            if error_m > 0.5:
                continue
            
            alpha_m = 0.5 * np.log((1 - error_m) / (error_m + 1e-10))
            base_models.append(graphsage)
            model_weights.append(alpha_m)

            weights *= np.exp(alpha_m * incorrect)
            weights /= np.sum(weights)

        with torch.no_grad():
            final_train_auc = roc_auc_score(labels[train_idx], boosting_predict(train_idx, base_models, model_weights, edge_index))
            final_val_auc = roc_auc_score(labels[val_idx], boosting_predict(val_idx, base_models, model_weights, edge_index))
            print(f"Final Train AUC: {final_train_auc:.4f}, Final Val AUC: {final_val_auc:.4f}")
            auc_scores.append(final_val_auc)
    
    avg_auc = sum(auc_scores) / len(auc_scores)
    print(f"Average Validation AUC: {avg_auc:.4f}")
    return avg_auc