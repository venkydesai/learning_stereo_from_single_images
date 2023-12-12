import torch
import torch.nn as nn
from torchvision import models

class PSMNetDecoder(nn.Module):
    def __init__(self):
        super(PSMNetDecoder, self).__init__()
        # Assuming input tensor shape is [batch_size, 152, 7, 7]
        
        # Upsampling layers
        self.upsample1 = nn.Conv2d(152, 128, kernel_size=3, padding=1)
        self.upsample2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # Pretrained VGG16 Network
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16_features = self.vgg16.features

    def forward(self, x):
        # Upsample the input
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upsample1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        x = self.upsample2(x)

        # Pass through VGG16
        x = self.vgg16_features(x)
        return x

# Example usage
decoder = PSMNetDecoder()
input_tensor = torch.randn(1, 152, 7, 7)  # Example input
output = decoder(input_tensor)
print(output.shape)
