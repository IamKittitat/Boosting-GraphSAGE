import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
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

def train_boosting_graphsage(features_train, features_val, adj_matrix_train, labels_train, labels_val, edge_index_train, 
                             neighbor_val, embed_dim, lr, base_estimators, num_epochs, num_layers):
    def boosting_predict(base_models, model_weights, features, edge_index):
        final_predictions = torch.zeros((len(features), 2))
        for model, weight in zip(base_models, model_weights):
            out_val = model(features, edge_index)
            weight = torch.tensor(weight, dtype=torch.float32)
            final_predictions += weight * out_val
        
        final_predictions = torch.softmax(final_predictions, dim=1)
        return final_predictions[:, 1]
    
    M = base_estimators
    base_models = []
    model_weights = []
    N = features_train.shape[0]
    num_feats = features_train.shape[1]
    weights = np.ones(N) / N
    
    for m in range(M):
        print(f"Base estimators: {m+1}/{M}")
        bootstrap_idx = np.random.choice(np.arange(N), size=N, replace=True, p=weights)
        graphsage = GraphSAGE(num_feats, embed_dim, 2, num_layers=num_layers, dropout=0, use_batchnorm=False)
        optimizer = torch.optim.Adam(graphsage.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(num_epochs):
            graphsage.train()
            optimizer.zero_grad()
            out = graphsage(features_train, edge_index_train)
            loss = criterion(out[bootstrap_idx], labels_train[bootstrap_idx])
            loss.backward()
            optimizer.step()

        graphsage.eval()
        with torch.no_grad():
            out = graphsage(features_train, edge_index_train)
            _, pred = out.max(dim=1)

        incorrect = np.array(pred[bootstrap_idx] != labels_train[bootstrap_idx]).astype(int)
        error_m = np.sum(weights * incorrect) / np.sum(weights)
        if error_m > 0.5:
            print(f"Error too high: {error_m:.4f} > 0.5, skipping this model")
            continue
        
        alpha_m = 0.5 * np.log((1 - error_m) / (error_m + 1e-10))
        base_models.append(graphsage)
        model_weights.append(alpha_m)

        incorrect[incorrect == 0] = -1
        weights *= np.exp(alpha_m * incorrect)
        weights /= np.sum(weights)

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
            probs = boosting_predict(base_models, model_weights, new_data.x, new_data.edge_index)[0].item()
            val_prob.append(probs)
    
    val_prob = torch.tensor(val_prob, dtype=torch.float32)
    final_train_auc = roc_auc_score(labels_train.cpu().numpy(), boosting_predict(base_models, model_weights, features_train, edge_index_train).detach().numpy())
    final_val_auc = roc_auc_score(labels_val.cpu().numpy(), val_prob.detach().numpy())
    print(f"Final Train AUC: {final_train_auc:.4f}, Final Val AUC: {final_val_auc:.4f}")

    return final_train_auc, final_val_auc

def train_gradient_boosting_graphsage(
    features_train, features_val, adj_matrix_train,
    labels_train, labels_val, edge_index_train, neighbor_val,
    embed_dim, lr, base_estimators, num_epochs, num_layers,
    learning_rate_boost=0.1
):
    sigmoid = torch.nn.Sigmoid()

    M = base_estimators
    base_models = []

    F_train = torch.zeros(len(features_train), dtype=torch.float32)

    for m in range(M):
        print(f"Boosting round: {m + 1}/{M}")
        p_train = sigmoid(F_train)

        residuals = (p_train - labels_train.float()).detach()

        graphsage = GraphSAGE(features_train.size(1), embed_dim, 1, num_layers=num_layers)
        optimizer = torch.optim.Adam(graphsage.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(num_epochs):
            graphsage.train()
            optimizer.zero_grad()
            out = graphsage(features_train, edge_index_train).squeeze()
            loss = criterion(out, residuals)
            loss.backward()
            optimizer.step()

        base_models.append(graphsage)

        graphsage.eval()
        with torch.no_grad():
            update = graphsage(features_train, edge_index_train).squeeze()
        F_train -= learning_rate_boost * update 

    final_train_probs = sigmoid(F_train)
    final_train_auc = roc_auc_score(labels_train.cpu().numpy(), final_train_probs.cpu().numpy())

    val_probs = []
    for features_node, neighbor_node in zip(features_val, neighbor_val):
        features_neighbor = features_train[neighbor_node]
        features_subgraph = torch.cat([features_node.unsqueeze(0), features_neighbor], dim=0)
        num_neighbors = len(neighbor_node)
        new_edges = []

        for i in range(1, num_neighbors + 1):
            new_edges.append([0, i])
            new_edges.append([i, 0])

        for i in range(num_neighbors):
            for j in range(i + 1, num_neighbors):
                if adj_matrix_train[neighbor_node[i]][neighbor_node[j]] == 1:
                    new_edges.append([i + 1, j + 1])
                    new_edges.append([j + 1, i + 1])

        edge_index_new = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
        F_val = torch.tensor(0.0)
        for model in base_models:
            model.eval()
            with torch.no_grad():
                out = model(features_subgraph, edge_index_new).squeeze()[0]
            F_val -= learning_rate_boost * out

        prob_val = sigmoid(F_val.clone().detach())
        val_probs.append(prob_val.item())

    final_val_auc = roc_auc_score(labels_val.cpu().numpy(), np.array(val_probs))
    print(f"Final Train AUC: {final_train_auc:.4f}, Final Val AUC: {final_val_auc:.4f}")

    return final_train_auc, final_val_auc
