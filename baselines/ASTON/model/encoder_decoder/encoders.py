import torch
from torch import nn

from model.GRNN import GRNN
from model.encoder_decoder.interfaces import Encoder
from model.encoder_decoder.utils import Time2Vec, PositionalEncoding
from model.graph_layers.GraphConv import GraphConv


class Seq2SeqEncoderResourceNoPositional(Encoder):
    """The RNN encoder for sequence to sequence learning."""

    def __init__(self, num_activities, num_resources, embed_size, num_hiddens, time_features, num_layers, N, F, vectorizer, adjacency_matrix,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoderResourceNoPositional, self).__init__(**kwargs)
        # Embedding layer
        self.embed_size = embed_size
        #self.time_features = time_features
        self.time_embedding = 16
        self.time2vecs = nn.ModuleList([Time2Vec(self.time_embedding) for _ in range(time_features)])
        self.embedding_activity = nn.Embedding(num_activities, embed_size)
        self.num_resources = num_resources
        # self.pos_encoding = PositionalEncoding(embed_size, dropout)
        grnn_out_size = 64
        if num_resources > 0:
            self.embedding_resource = nn.Embedding(num_resources, embed_size)
            self.rnn = nn.GRU(embed_size + embed_size + (time_features * (self.time_embedding + 1)) + grnn_out_size, num_hiddens, num_layers,
                              dropout=dropout)
        else:
            self.rnn = nn.GRU(embed_size + (time_features * (self.time_embedding + 1)) + grnn_out_size, num_hiddens, num_layers,
                              dropout=dropout)

        self.grnn = GRNN(N, F, vectorizer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.adjacency_matrix = torch.tensor(adjacency_matrix).to(self.device)


    def forward(self, X, X_grnn, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X_out_grnn = self.grnn(X_grnn, self.adjacency_matrix)

        X_emb_activity = self.embedding_activity(X[:, :, 0].to(torch.int))
        if self.num_resources > 0:
            X_emb_resource = self.embedding_resource(X[:, :, 1].to(torch.int))
            features = [X_emb_activity, X_emb_resource]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 2:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)

            # X_emb = self.pos_encoding(self.embedding(X[:,:,0].to(torch.int)) * math.sqrt(self.embed_size))
            # In RNN models, the first axis corresponds to time steps
            X = torch.cat(features, 2).permute(1, 0, 2)
        else:
            features = [X_emb_activity]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 1:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)
            X = torch.cat(features, 2).permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros

        X = torch.cat((X, X_out_grnn), dim=2)

        output, state = self.rnn(X)

        # DEACTIVATE POSITIONAL ENCODING
        # state = self.pos_encoding(state * math.sqrt(self.embed_size))

        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class EncoderGRNNHiddenState(Encoder):
    def __init__(self, num_activities, num_resources, embed_size, num_hiddens, time_features, num_layers, N, F, vectorizer, adjacency_matrix,
                 dropout=0, **kwargs):
        super(EncoderGRNNHiddenState, self).__init__(**kwargs)
        # Embedding layer
        self.embed_size = embed_size
        #self.time_features = time_features
        self.time_embedding = 16
        self.time2vecs = nn.ModuleList([Time2Vec(self.time_embedding) for _ in range(time_features)])
        self.embedding_activity = nn.Embedding(num_activities, embed_size)
        self.num_resources = num_resources
        # self.pos_encoding = PositionalEncoding(embed_size, dropout)
        grnn_out_size = 64
        if num_resources > 0:
            self.embedding_resource = nn.Embedding(num_resources, embed_size)
            #self.rnn = nn.GRU(embed_size + embed_size + (time_features * (self.time_embedding + 1)) + grnn_out_size, num_hiddens, num_layers,
            #                  dropout=dropout)
            self.rnn = nn.GRU(embed_size + embed_size + (time_features * (self.time_embedding + 1)), num_hiddens, num_layers,
                              dropout=dropout)
        else:
            #self.rnn = nn.GRU(embed_size + (time_features * (self.time_embedding + 1)) + grnn_out_size, num_hiddens, num_layers,
            #                  dropout=dropout)
            self.rnn = nn.GRU(embed_size + (time_features * (self.time_embedding + 1)), num_hiddens, num_layers,
                             dropout=dropout)

        self.grnn = GRNN(N, F, vectorizer)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.adjacency_matrix = torch.tensor(adjacency_matrix).to(self.device)


    def forward(self, X, X_grnn, *args):
        # The output `X` shape: (`batch_size`, `num_steps`, `embed_size`)
        X_out_grnn, X_out_grnn_first = self.grnn(X_grnn, self.adjacency_matrix, return_every_state=True)

        X_emb_activity = self.embedding_activity(X[:, :, 0].to(torch.int))
        if self.num_resources > 0:
            X_emb_resource = self.embedding_resource(X[:, :, 1].to(torch.int))
            features = [X_emb_activity, X_emb_resource]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 2:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)

            # X_emb = self.pos_encoding(self.embedding(X[:,:,0].to(torch.int)) * math.sqrt(self.embed_size))
            # In RNN models, the first axis corresponds to time steps
            X = torch.cat(features, 2).permute(1, 0, 2)
        else:
            features = [X_emb_activity]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 1:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)
            X = torch.cat(features, 2).permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros

        max_pool_2d = torch.nn.MaxPool2d(kernel_size=(X_out_grnn.shape[1], X_out_grnn.shape[2]))
        X_out_grnn_first = X_out_grnn_first.permute(0, 3, 1, 2)
        X_out_grnn = X_out_grnn.permute(0, 3, 1, 2)
        first_state = max_pool_2d(X_out_grnn_first).squeeze(-1).squeeze(-1).unsqueeze(0)
        second_state = max_pool_2d(X_out_grnn).squeeze(-1).squeeze(-1).unsqueeze(0)
        grnn_states = torch.cat((first_state, second_state), dim=0)

        X_out_grnn = X_out_grnn.permute(0, 2, 3, 1)
        X_out_grnn, _ = torch.max(X_out_grnn, dim=2)
        X_out_grnn = X_out_grnn.permute(1, 0, 2)

        output, state = self.rnn(X)

        state = torch.cat((state, grnn_states), dim=2)
        output = torch.cat((output, X_out_grnn), dim=2)

        # DEACTIVATE POSITIONAL ENCODING
        # state = self.pos_encoding(state * math.sqrt(self.embed_size))

        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, state

class Seq2SeqSimpleTransformerEncoderOnlyActivities(Encoder):
    """The RNN encoder for sequence to sequence learning."""

    def __init__(self, num_activities, num_resources, embed_size, num_hiddens, time_features, num_layers, N, F, vectorizer, adjacency_matrix,
                 dropout=0, **kwargs):

        self.eoc_token = num_activities - 1

        super(Seq2SeqSimpleTransformerEncoderOnlyActivities, self).__init__(**kwargs)
        # Embedding layer
        self.embed_size = embed_size
        #self.embed_size = 512
        self.time_embedding = 16
        #self.time2vecs = nn.ModuleList([Time2Vec(self.time_embedding) for _ in range(time_features)])
        self.embedding_activity = nn.Embedding(num_activities, embed_size)
        self.num_resources = num_resources
        # self.pos_encoding = PositionalEncoding(embed_size, dropout)
        if num_resources > 0:
            #dimensions = embed_size + embed_size + (time_features * (self.time_embedding + 1))
            dimensions = embed_size
            #self.embedding_resource = nn.Embedding(num_resources, embed_size)
        else:
            #dimensions = embed_size + (time_features * (self.time_embedding + 1))
            dimensions = embed_size

        #hidden = 64
        hidden = embed_size
        self.positional_encoding = PositionalEncoding(num_hiddens=dimensions, dropout=0)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=8)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        #self.project = torch.nn.LazyLinear(hidden)



    def forward(self, X, X_grnn, *args):
        X_emb_activity = self.embedding_activity(X[:, :, 0].to(torch.int))
        """
        if self.num_resources > 0:
            X_emb_resource = self.embedding_resource(X[:, :, 1].to(torch.int))
            features = [X_emb_activity, X_emb_resource]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 2:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)

            # X_emb = self.pos_encoding(self.embedding(X[:,:,0].to(torch.int)) * math.sqrt(self.embed_size))
            # In RNN models, the first axis corresponds to time steps
            X = torch.cat(features, 2)
        else:
            X = torch.cat((X_emb_activity, X[:, :, 1:]), 2)
        # When state is not mentioned, it defaults to zeros

        X = self.positional_encoding(X)

        X = self.project(X)
        """

        # Pad the zeros
        mask = (X[:, :, 0] != 0).float()
        mask = mask.permute(1, 0)

        X = self.positional_encoding(X_emb_activity)

        output = self.encoder(X, src_key_padding_mask=mask)


        return output, None

class Seq2SeqSimpleTransformerEncoder(Encoder):
    """The RNN encoder for sequence to sequence learning."""

    def __init__(self, num_activities, num_resources, embed_size, num_hiddens, time_features, num_layers, N, F, vectorizer,
                 dropout=0, **kwargs):

        self.eoc_token = num_activities - 1

        super(Seq2SeqSimpleTransformerEncoder, self).__init__(**kwargs)
        # Embedding layer
        self.embed_size = embed_size
        #self.embed_size = 512
        self.time_embedding = 16
        self.time2vecs = nn.ModuleList([Time2Vec(self.time_embedding) for _ in range(time_features)])
        self.embedding_activity = nn.Embedding(num_activities, embed_size)
        self.num_resources = num_resources
        # self.pos_encoding = PositionalEncoding(embed_size, dropout)
        if num_resources > 0:
            dimensions = embed_size + embed_size + (time_features * (self.time_embedding + 1))
            #dimensions = embed_size
            self.embedding_resource = nn.Embedding(num_resources, embed_size)
        else:
            dimensions = embed_size + (time_features * (self.time_embedding + 1))
            #dimensions = embed_size

        #hidden = 64
        hidden = self.embed_size
        self.positional_encoding = PositionalEncoding(num_hiddens=dimensions, dropout=0)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=8)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.project = torch.nn.LazyLinear(hidden)



    def forward(self, X, X_grnn, A, *args):
        # Pad the zeros
        mask = (X[:, :, 0] != 0).float()
        mask = mask.permute(1, 0)

        X_emb_activity = self.embedding_activity(X[:, :, 0].to(torch.int))
        if self.num_resources > 0:
            X_emb_resource = self.embedding_resource(X[:, :, 1].to(torch.int))
            features = [X_emb_activity, X_emb_resource]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 2:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)

            # X_emb = self.pos_encoding(self.embedding(X[:,:,0].to(torch.int)) * math.sqrt(self.embed_size))
            # In RNN models, the first axis corresponds to time steps
            X = torch.cat(features, 2)
        else:
            X = torch.cat((X_emb_activity, X[:, :, 1:]), 2)
        # When state is not mentioned, it defaults to zeros

        X = self.positional_encoding(X)

        X = self.project(X)

        #X = self.positional_encoding(X_emb_activity)

        output = self.encoder(X, src_key_padding_mask=mask)

        return output, None


class Seq2SeqEncoderGRNNDualLSTM(Encoder):
    def __init__(self, num_activities, num_resources, embed_size, num_hiddens, time_features, num_layers, N, F, vectorizer,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoderGRNNDualLSTM, self).__init__(**kwargs)
        self.embed_size = embed_size
        self.time_embedding = 16
        self.time2vecs = nn.ModuleList([Time2Vec(self.time_embedding) for _ in range(time_features)])
        self.embedding_activity = nn.Embedding(num_activities, embed_size)
        self.num_resources = num_resources
        grnn_out_size = 64
        if num_resources > 0:
            self.embedding_resource = nn.Embedding(num_resources, embed_size)
            self.rnn = nn.GRU(embed_size + embed_size + (time_features * (self.time_embedding + 1)), num_hiddens, num_layers,
                              dropout=dropout)
        else:
            self.rnn = nn.GRU(embed_size + time_features, num_hiddens, num_layers,
                              dropout=dropout)

        self.dual_lstm_1 = nn.LSTM(F, num_hiddens, num_layers, dropout=dropout, batch_first=True) # This lstm can project to any dimension
        self.dual_lstm_2 = nn.LSTM(N, N, num_layers, dropout=dropout, batch_first=True) # This lstm must project to the node dimension
        self.gcn_list = torch.nn.ModuleList()
        for i in range(num_layers):
            self.gcn_list.append(GraphConv(num_hiddens, N, num_hiddens + F))


    def forward(self, X, X_grnn, A, *args):
        # First unpack through the node dimension and apply the first dual lstm
        concat_feature_lstm = []
        for x_f in torch.unbind(X_grnn, dim=2):
            _, (x_f, _) = self.dual_lstm_1(x_f.float())
            concat_feature_lstm.append(x_f.unsqueeze(2))
        concat_feature_lstm = torch.cat(concat_feature_lstm, dim=2)

        # Then unpack through the feature dimension and apply the second dual lstm
        concat_node_lstm = []
        for x_n in torch.unbind(X_grnn, dim=3):
            _, (x_n, _) = self.dual_lstm_2(x_n.float())
            concat_node_lstm.append(x_n.unsqueeze(2))
        concat_node_lstm = torch.cat(concat_node_lstm, dim=2).permute(0, 1, 3, 2)
        X_out_grnn = torch.cat((concat_feature_lstm, concat_node_lstm), dim=-1)
        X_grnn_list = []
        for i, X_grnn in enumerate(torch.unbind(X_out_grnn, dim=0)):
            X_grnn = self.gcn_list[i]([X_grnn, A.float()])
            X_grnn , _ = torch.max(X_grnn, dim=1)
            X_grnn_list.append(X_grnn.unsqueeze(0))
        X_grnn_list = torch.cat(X_grnn_list, dim=0)


        X_emb_activity = self.embedding_activity(X[:, :, 0].to(torch.int))
        if self.num_resources > 0:
            X_emb_resource = self.embedding_resource(X[:, :, 1].to(torch.int))
            features = [X_emb_activity, X_emb_resource]
            for i, time_feature in enumerate(torch.unbind(X[:, :, 2:], dim=-1)):
                X_emb_time = self.time2vecs[i](time_feature)
                features.append(X_emb_time)

            # X_emb = self.pos_encoding(self.embedding(X[:,:,0].to(torch.int)) * math.sqrt(self.embed_size))
            # In RNN models, the first axis corresponds to time steps
            X = torch.cat(features, 2).permute(1, 0, 2)
        else:
            X = torch.cat((X_emb_activity, X[:, :, 1:]), 2).permute(1, 0, 2)
        # When state is not mentioned, it defaults to zeros

        #X = torch.cat((X, X_out_grnn), dim=2)

        output, state = self.rnn(X)

        # Duplicate the outputs of the gcn operation so that it is consistent with the hidden state of the rnn

        # DEACTIVATE POSITIONAL ENCODING
        # state = self.pos_encoding(state * math.sqrt(self.embed_size))

        # `output` shape: (`num_steps`, `batch_size`, `num_hiddens`)
        # `state` shape: (`num_layers`, `batch_size`, `num_hiddens`)
        return output, torch.cat((state, X_grnn_list), dim=2)
