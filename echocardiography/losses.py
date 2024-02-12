import torch
import torch.nn as nn

## define weighted MSE loss
class WeightedMSELoss(nn.Module):
    def __init__(self, device):
        super(WeightedMSELoss, self).__init__()
        self.device = device

    def forward(self, output, label):
        weight = torch.ones(label.shape).to(self.device)
        ## binary the targert with 0.5
        mask = (label > 0.5).float().to(self.device)
        ## give me the total numer of 0 and 1
        num_0 = torch.sum(mask == 0).float().to(self.device)
        num_1 = torch.sum(mask == 1).float().to(self.device)
        ## give me the frequency of 0 and 1
        num_0 = num_0 / (num_0 + num_1).to(self.device)
        num_1 = num_1 / (num_0 + num_1).to(self.device)

        # create a weight tensor substituting 1 with 1/num_1 and 0 with 1/num_0
        weight = torch.where(mask == 1, 1/num_1, weight).to(self.device)
        weight = torch.where(mask == 0, 1/num_0, weight).to(self.device)
        print(f'weight, {weight}')
        return torch.mean(weight * (label - output) ** 2)

class RMSELoss(torch.nn.Module):
    """
    Root Mean Square Error Loss
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,label,target):
        loss = torch.sqrt(self.mse(label,target) + self.eps)
        return loss

class WeightedRMSELoss(torch.nn.Module):
    """
    Weighted Root Mean Square Error Loss
    """
    def __init__(self, device, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        self.device = device
        
    def forward(self,output,label):
        weight = torch.ones(label.shape).to(self.device)
        ## binary the targert with 0.5
        mask = (label > 0.5).float().to(self.device)
        ## give me the total numer of 0 and 1
        num_0 = torch.sum(mask == 0).float().to(self.device)
        num_1 = torch.sum(mask == 1).float().to(self.device)
        ## give me the frequency of 0 and 1
        num_0 = num_0 / (num_0 + num_1).to(self.device)
        num_1 = num_1 / (num_0 + num_1).to(self.device)

        # create a weight tensor substituting 1 with 1/num_1 and 0 with 1/num_0
        weight = torch.where(mask == 1, 1/num_1, weight).to(self.device)
        weight = torch.where(mask == 0, 1/num_0, weight).to(self.device)
        loss = torch.sqrt((torch.mean(weight * (label - output) ** 2)) + self.eps)
        return loss

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loss = torch.nn.MSELoss()
    loss = nn.MSELoss()
    output = torch.tensor([[0.1,0.2],[0.3,0.4]]).to(device)
    label = torch.tensor([[0.1,0.1],[0.6,0.1]]).to(device)
    output = loss(output, label)
    w_mse = WeightedMSELoss(device)(output, label)
    w_rmse = WeightedRMSELoss(device)(output, label)
    rmse = RMSELoss()(output, label)

    print(f'W_MSE: {w_mse}')
    print(f'W_RMSE: {w_rmse}')
    print(f'RMSE: {rmse}')