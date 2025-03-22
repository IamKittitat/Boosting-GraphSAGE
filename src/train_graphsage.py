import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    def boosting_predict(val_idx, base_models, model_weights, edge_index):
        final_predictions = torch.zeros((len(val_idx), config['model']['num_classes']))
        for model, weight in zip(base_models, model_weights):
            out_val = model(features, edge_index)
            weight = torch.tensor(weight, dtype=torch.float32)
            probs = torch.softmax(out_val[val_idx], dim=1)
            final_predictions += weight * probs
        return final_predictions[:, 1]
    
    M = config['model']['base_estimators']    
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
            graphsage = GraphSAGE(num_feats, config['model']['embed_dim'], 2)
            optimizer = torch.optim.Adam(graphsage.parameters(), lr=config['model']['lr'])
            criterion = nn.CrossEntropyLoss()

            for _ in range(config['model']['epoch']):
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
            print(f"Final Train AUC: {final_train_auc:.4f}, Final Test AUC: {final_val_auc:.4f}")
            auc_scores.append(final_val_auc)
    
    avg_auc = sum(auc_scores) / len(auc_scores)
    print(f"Average Validation AUC: {avg_auc:.4f}")