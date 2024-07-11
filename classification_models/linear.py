import torch


class LogReg(torch.nn.Module):
    """
    A classic logistic regression with unnormalized log-probabilites at the output.
    """
    
    def __init__(self, input_dim, num_classes):
        super(LogReg, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(self.input_dim, self.num_classes)
    
    def forward(self, x):
        x = self.linear(x)
        return x
