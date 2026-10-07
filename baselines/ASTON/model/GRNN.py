import torch

from model.layers.GGRNN import GGRNN


class GRNN(torch.nn.Module):

    def __init__(self, N, F, vectorizer):
        super().__init__()
        self.N = N
        self.F = F

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.embedded_F = 32 * self.F
        self.GGRNN_1 = GGRNN(64, self.N, F, dropout=0, return_sequences=True,
                             bidirectional=False, n_layer=1).to(self.device)
        self.GGRNN_2 = GGRNN(64, self.N, 64, dropout=0, return_sequences=True,
                             bidirectional=False, n_layer=1).to(self.device)


        #self.node_embedding = torch.nn.Embedding(N + 2, 32, padding_idx=0).to(self.device)
        #self.transition_embedding = torch.nn.Embedding(len(vectorizer.transitions) + 2, 32, padding_idx=0).to(self.device)

    def forward(self, X_grnn, A_in, return_every_state=False):

        X_1 = self.GGRNN_1([X_grnn.float(), A_in.float()])

        X = self.GGRNN_2([X_1, A_in.float()])

        # Perform the pooling
        """
        mini_batch_pooled = []
        for mini_batch in torch.unbind(X, dim=0):
            pooled_trace = []
            for activation_idx, graph_element in enumerate(reversed(torch.unbind(mini_batch, dim=0)), start=1):
                pooled_trace.insert(0, torch.unsqueeze(torch.max(graph_element, dim=0)[0], dim=0))

            pooled_trace = torch.cat(pooled_trace, dim=0)

            mini_batch_pooled.append(torch.unsqueeze(pooled_trace, dim=0))
        X = torch.cat(mini_batch_pooled, dim=0)
        """
        #X = X[:, -1, :]

        if not return_every_state:
            X, _ = torch.max(X, dim=2)

            X = X.permute(1, 0, 2)
            return X
        else:
            return X, X_1
