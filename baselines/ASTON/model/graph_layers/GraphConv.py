import torch


class GraphConv(torch.nn.Module):
    def __init__(self, channels, N, F):
        super().__init__()
        self.N = N
        self.channels = channels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.kernel = torch.nn.Parameter(data=torch.Tensor(F, channels), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.kernel)
        #torch.nn.init.kaiming_uniform_(self.kernel)

        self.bias = torch.nn.Parameter(data=torch.Tensor(N, channels), requires_grad=True)
        torch.nn.init.ones_(self.bias)

        #self.linear = torch.nn.Linear(F, channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        X = x[0]
        A = x[1]

        out = torch.matmul(X, self.kernel)

        #out = self.linear(X)

        out = torch.matmul(A, out)

        #expand_bias = self.bias.expand(self.bias, self.N, self.channels)
        out = out + self.bias

        #output = self.relu(out)
        output = out

        return output
