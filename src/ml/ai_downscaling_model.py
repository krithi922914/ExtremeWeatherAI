import torch
import torch.nn as nn


class WeatherDownscalingCNN(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature extraction
        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU()
        )


        # Convert learned features into a
        # higher-resolution weather field
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )


    def forward(self, x):

        x = self.encoder(x)

        x = self.decoder(x)

        return x


if __name__ == "__main__":

    print("AI Weather Downscaling CNN")
    print("=" * 40)

    model = WeatherDownscalingCNN()

    print(model)

    # Test with a 12 km weather grid
    sample_input = torch.randn(
        1,
        1,
        10,
        10
    )

    output = model(sample_input)

    print()
    print("Input shape :", sample_input.shape)
    print("Output shape:", output.shape)

    print()
    print("AI downscaling model created successfully!")