import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import dense_to_sparse
from src.graphsage.model import GraphSAGE

def train_graphsage(features, adj_matrix, labels, config):    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []

    num_feats = features.shape[1]
    edge_index, _ = dense_to_sparse(adj_matrix)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(features)), labels)):
        print(f"Fold {fold+1}")
        train_idx = torch.tensor(train_idx)
        val_idx = torch.tensor(val_idx)
        
        graphsage = GraphSAGE(num_feats, config['model']['embed_dim'], 2)
        optimizer = torch.optim.Adam(graphsage.parameters(), lr=config['model']['lr'])
        criterion = nn.CrossEntropyLoss()

        for epoch in range(config['model']['epoch']):
            graphsage.train()
            optimizer.zero_grad()
            out = graphsage(features, edge_index)
            loss = criterion(out[train_idx], labels[train_idx])
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                graphsage.eval()
                with torch.no_grad():
                    out_val = graphsage(features, edge_index)
                    probs = torch.softmax(out_val[val_idx], dim=1)[:, 1]
                    auc = roc_auc_score(labels[val_idx].cpu().numpy(), probs.cpu().numpy())
                print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Val AUC: {auc:.4f}")

        # Evaluate on test indices for the current fold
        graphsage.eval()
        with torch.no_grad():
            out_val = graphsage(features, edge_index)
            probs = torch.softmax(out_val[val_idx], dim=1)[:, 1]
            fold_auc = roc_auc_score(labels[val_idx].cpu().numpy(), probs.cpu().numpy())
            auc_scores.append(fold_auc)
            print(f"Fold {fold+1} Validation AUC: {fold_auc:.4f}\n")

    avg_auc = sum(auc_scores) / len(auc_scores)
    print(f"Average Validation AUC: {avg_auc:.4f}")

def train_boosting_graphsage(features, adj_matrix, labels, config):
    def boosting_predict(val_indices):
        final_predictions = np.zeros((len(val_indices), config['model']['num_classes']))
        for model, weight in zip(base_models, model_weights):
            predictions = model(val_indices).detach().numpy()
            final_predictions += weight * predictions
        return final_predictions[:, 1]
    
    M = config['model']['base_estimators']    
    base_models = []
    model_weights = []
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []
    train_auc_scores = []
    
    for train_idx, val_idx in skf.split(np.arange(len(features)), labels):
        N = len(train_idx)
        weights = np.ones(N) / N 
        
        for m in range(M):
            print(f"Base estimators: {m+1}/{M}")
            bootstrap_indices = np.random.choice(train_idx, size=N, replace=True, p=weights)
            bootstrap_features = features[bootstrap_indices]
            bootstrap_adj = adj_matrix[bootstrap_indices][:, bootstrap_indices]
            bootstrap_labels = labels[bootstrap_indices]
    
            model = SupervisedGraphSage(features=features, adj_matrix=adj_matrix,
                                        num_classes=config['model']['num_classes'],
                                        num_layers=config['model']['num_layers'],
                                        num_sample1=config['model']['num_sample1'],
                                        num_sample2=config['model']['num_sample2'],
                                        embed_dim=config['model']['embed_dim'])
            optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['lr'])
            
            for epoch in range(config['model']['epoch']):
                optimizer.zero_grad()
                loss = model.loss(np.arange(N), torch.LongTensor(bootstrap_labels))
                loss.backward()
                optimizer.step()
            
            train_preds = model(np.arange(N)).detach().numpy().argmax(axis=1)
            incorrect = (train_preds != bootstrap_labels)
            error_m = np.sum(weights * incorrect) / np.sum(weights)
            if error_m > 0.5:
                print(error_m)
                continue
            
            alpha_m = 0.5 * np.log((1 - error_m) / (error_m + 1e-10))
            base_models.append(model)
            model_weights.append(alpha_m)

            train_auc = roc_auc_score(labels[train_idx], boosting_predict(train_idx))
            print(f"Train AUC:", train_auc)
            train_auc_scores.append(train_auc)
            
            weights *= np.exp(alpha_m * incorrect)
            weights /= np.sum(weights)
        
        val_auc = roc_auc_score(labels[val_idx], boosting_predict(val_idx))
        auc_scores.append(val_auc)
        print(f"Validation AUC:", val_auc)
    
    print("Average Train AUC:", sum(train_auc_scores) / len(train_auc_scores))
    avg_auc = sum(auc_scores) / len(auc_scores)
    print("Average Validation AUC:", avg_auc)
    
    return avg_auc
