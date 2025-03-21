import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold

from src.graphsage.model import SupervisedGraphSage

def train_graphsage(features, adj_matrix, labels, config):
    graphsage = SupervisedGraphSage(features=features, 
                                    adj_matrix=adj_matrix, 
                                    num_classes=config['model']['num_classes'], 
                                    num_layers=config['model']['num_layers'], 
                                    num_sample1=config['model']['num_sample1'], 
                                    num_sample2=config['model']['num_sample2'], 
                                    embed_dim=config['model']['embed_dim']
                                    )
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_scores = []
    
    for train_idx, val_idx in skf.split(np.arange(len(features)), labels):
        optimizer = torch.optim.Adam(graphsage.parameters(), lr=config['model']['lr'])
        
        for epoch in range(config['model']['epoch']):
            random.shuffle(train_idx)
            optimizer.zero_grad()
            loss = graphsage.loss(train_idx, torch.LongTensor(labels[np.array(train_idx)]))
            loss.backward()
            optimizer.step()
        
        val_output = graphsage(val_idx).detach().numpy()
        val_auc = roc_auc_score(labels[val_idx], val_output[:, 1])
        auc_scores.append(val_auc)
        print(f"Validation AUC:", val_auc)
    
    avg_auc = sum(auc_scores) / len(auc_scores)
    print("Average Validation AUC:", avg_auc)
    
    return avg_auc

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
