import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data

features = torch.randn(1000, 16) 
labels = torch.randint(0, 2, (1000,))
adj_matrix = torch.randint(0, 2, (1000, 1000)).float()
train_idx = torch.arange(0, 800)
test_idx = torch.arange(800, 1000)

edge_index, _ = dense_to_sparse(adj_matrix)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

in_channels = features.shape[1]
hidden_channels = 64
num_classes = 2

model = GraphSAGE(in_channels, hidden_channels, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

num_epochs = 50
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()
    out = model(features, edge_index)
    loss = criterion(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        model.eval()
        _, pred = out.max(dim=1)
        train_correct = pred[train_idx].eq(labels[train_idx]).sum().item()
        test_correct = pred[test_idx].eq(labels[test_idx]).sum().item()
        train_acc = train_correct / len(train_idx)
        test_acc = test_correct / len(test_idx)
        print(f"Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")


# --- New Nodes ---
new_node_feature = torch.randn(1, features.shape[1])
neighbor_features = torch.randn(5, features.shape[1])
new_subgraph_features = torch.cat([new_node_feature, neighbor_features], dim=0)

# --- New subgraph connectivity ---
# Build the edge_index for the new subgraph: assume the new node is connected to all neighbors (undirected graph).
num_neighbors = neighbor_features.shape[0]
new_edges = []
for i in range(1, num_neighbors + 1):
    new_edges.append([0, i])
    new_edges.append([i, 0])
# For demo, assume no neighbor-neighbor connections. --> Please add here using adj_matrix

edge_index_new = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
new_data = Data(x=new_subgraph_features, edge_index=edge_index_new)

model.eval()
with torch.no_grad():
    out = model(new_data.x, new_data.edge_index)
    new_node_logits = out[0]
    predicted_class = new_node_logits.argmax(dim=0).item()

print(f"Predicted class for the new node: {predicted_class}")