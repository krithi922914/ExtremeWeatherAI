import torch
import torch.nn as nn


class GraphLayer(nn.Module):

    def __init__(self, input_features, output_features):

        super().__init__()

        self.self_layer = nn.Linear(
            input_features,
            output_features
        )

        self.neighbor_layer = nn.Linear(
            input_features,
            output_features
        )

        self.activation = nn.ReLU()

    def forward(self, x, edge_index):

        source_nodes = edge_index[0]
        target_nodes = edge_index[1]

        neighbor_messages = torch.zeros_like(
            self.self_layer(x)
        )

        transformed_neighbors = self.neighbor_layer(
            x[source_nodes]
        )

        neighbor_messages.index_add_(
            0,
            target_nodes,
            transformed_neighbors
        )

        output = (
            self.self_layer(x)
            + neighbor_messages
        )

        return self.activation(output)


class SpatioTemporalWeatherGNN(nn.Module):

    def __init__(
        self,
        input_features=4,
        hidden_features=32,
        output_features=2
    ):

        super().__init__()

        self.graph_layer1 = GraphLayer(
            input_features,
            hidden_features
        )

        self.graph_layer2 = GraphLayer(
            hidden_features,
            hidden_features
        )

        self.output_layer = nn.Linear(
            hidden_features,
            output_features
        )

    def forward(self, x, edge_index):

        x = self.graph_layer1(
            x,
            edge_index
        )

        x = self.graph_layer2(
            x,
            edge_index
        )

        graph_representation = torch.mean(
            x,
            dim=0,
            keepdim=True
        )

        output = self.output_layer(
            graph_representation
        )

        return output


if __name__ == "__main__":

    print("Spatio-Temporal Weather GNN")
    print("=" * 45)

    model = SpatioTemporalWeatherGNN()

    sample_nodes = torch.tensor(
        [
            [12.0, 82.0, 60.0, 85.0],
            [12.8, 82.8, 55.0, 79.0]
        ],
        dtype=torch.float32
    )

    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0]
        ],
        dtype=torch.long
    ).t()

    prediction = model(
        sample_nodes,
        edge_index
    )

    print("Node feature shape:", sample_nodes.shape)
    print("Edge shape:", edge_index.shape)
    print("Prediction shape:", prediction.shape)
    print("Predicted next location:", prediction)
