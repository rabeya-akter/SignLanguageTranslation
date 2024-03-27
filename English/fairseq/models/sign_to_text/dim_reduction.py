import torch
import torch.nn as nn

# Create a custom module that reduces the feature dimension from 512 to 256
class DimensionReductionLayerLinear(nn.Module):
    def __init__(self,input_dim=512,output_dim=256):
        super(DimensionReductionLayerLinear, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        # Reshape the input to [T * B, 512] for the linear layer
        #print('x:',x)
        T,B,C=x.shape[0],x.shape[1],x.shape[2]
        x = x.view(-1, 512)
        x = self.fc(x)
        # Reshape the output back to [T, B, 256]
        x = x.view(T, -1, 256)

        return x
    
class DimensionReductionLayerLSTM(nn.Module):
    def __init__(self,input_dim=512,output_dim=256):
        super(DimensionReductionLayerLSTM, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.output_dim, num_layers=1, batch_first=False)

        #self.fc = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        # Reshape the input to [T * B, 512] for the linear layer
        #print('x:',x)
        T,B,C=x.shape[0],x.shape[1],x.shape[2]
        x, _ = self.lstm(x)
        # Reshape the output back to [T, B, 256]
        #x = x.view(T, -1, 256)

        return x


