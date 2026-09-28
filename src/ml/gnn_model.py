import torch
import torch.nn as nn


class WeatherGNN(nn.Module):

    def __init__(self, input_features=4, hidden_features=32, output_features=2):

        super().__init__()

        self.layer1 = nn.Linear(
            input_features,
            hidden_features
        )

        self.layer2 = nn.Linear(
            hidden_features,
            hidden_features
        )

        self.output_layer = nn.Linear(
            hidden_features,
            output_features
        )

        self.activation = nn.ReLU()

    def forward(self, x):

        x = self.layer1(x)
        x = self.activation(x)

        x = self.layer2(x)
        x = self.activation(x)

        x = torch.mean(x, dim=1)

        output = self.output_layer(x)

        return output


if __name__ == "__main__":

    print("Weather GNN Model")
    print("=" * 30)

    model = WeatherGNN()

    print(model)

    sample_input = torch.randn(1, 2, 4)

    prediction = model(sample_input)

    print()
    print("Input shape:", sample_input.shape)
    print("Prediction shape:", prediction.shape)
    print("Predicted location:", prediction)
