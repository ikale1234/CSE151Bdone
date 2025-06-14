import torch
import torch.nn as nn
from omegaconf import DictConfig 
import torch.nn.functional as F 

def get_model(cfg: DictConfig):
    model_kwargs = {k: v for k, v in cfg.model.items() if k != "type"}
    model_kwargs["n_input_channels"] = len(cfg.data.input_vars)
    model_kwargs["n_output_channels"] = len(cfg.data.output_vars)



    if cfg.model.type == "simple_cnn":
        model = SimpleCNN(**model_kwargs)
    elif cfg.model.type == "conv_lstm": # New model type
        model = ConvLSTM(**model_kwargs)

    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")
    return model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.skip(identity)
        out = self.relu(out)

        return out


class SimpleCNN(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=64,
        depth=4,
        dropout_rate=0.2,
    ):
        super().__init__()

        # Initial convolution to expand channels
        self.initial = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        # Residual blocks with increasing feature dimensions
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim

        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(ResidualBlock(current_dim, out_dim))
            if i < depth - 1:  # Don't double the final layer
                current_dim *= 2

        # Final prediction layers
        self.dropout = nn.Dropout2d(dropout_rate)
        self.final = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(current_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, n_output_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.initial(x)

        for res_block in self.res_blocks:
            x = res_block(x)
        
        # MAYBE ADD LSTM PASS HERE

        x = self.dropout(x)
        x = self.final(x)

        return x




class ConvLSTM(nn.Module):
    def __init__(
        self,
        n_input_channels,
        n_output_channels,
        kernel_size=3,
        init_dim=48,
        depth=2,
        dropout_rate=0.2,
        lstm_hidden_dim=64, 
        num_lstm_layers=1,  
        grid_height=48,     
        grid_width=72,      
    ):
        super().__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.grid_height = grid_height
        self.grid_width = grid_width

        
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(init_dim),
            nn.ReLU(inplace=True),
        )

        current_dim = init_dim
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim 
            self.encoder.add_module(f'res_block_{i}', ResidualBlock(current_dim, out_dim))
            if i < depth - 1:
                current_dim *= 2

     
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                in_channels=(current_dim if i == 0 else lstm_hidden_dim),
                out_channels=lstm_hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                activation=nn.Tanh(),
                frame_size=(grid_height, grid_width) 
            ) for i in range(num_lstm_layers)
        ])


        self.decoder = nn.Sequential(
            nn.Conv2d(lstm_hidden_dim, lstm_hidden_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(lstm_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(lstm_hidden_dim // 2, n_output_channels, kernel_size=1),
        )

        self.dropout = nn.Dropout2d(dropout_rate) 

    def forward(self, x_sequence):

        batch_size, seq_len, _, H, W = x_sequence.size()

 
        hidden_states = []
        cell_states = []
        for i in range(self.num_lstm_layers):
            h, c = self.convlstm_cells[i].init_hidden(batch_size, (H, W))
            hidden_states.append(h)
            cell_states.append(c)


        output_sequence = []

        for t in range(seq_len):
            input_grid_t = x_sequence[:, t, :, :, :] 
            encoded_features_t = self.dropout(self.encoder(input_grid_t))
            current_input_for_lstm = encoded_features_t

            for i in range(self.num_lstm_layers):
                h, c = self.convlstm_cells[i](current_input_for_lstm, (hidden_states[i], cell_states[i]))
                hidden_states[i] = h 
                cell_states[i] = c   
                current_input_for_lstm = h 

            decoded_output_t = self.decoder(self.dropout(h))
            output_sequence.append(decoded_output_t)

        return decoded_output_t






class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        super(ConvLSTMCell, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.frame_size = frame_size 


        self.conv_i = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)
        self.conv_f = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)
        self.conv_o = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)
        self.conv_g = nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=self.kernel_size, padding=self.padding)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1) 

        i = torch.sigmoid(self.conv_i(combined))
        f = torch.sigmoid(self.conv_f(combined))
        o = torch.sigmoid(self.conv_o(combined))
        g = self.activation(self.conv_g(combined))

        c_next = f * c_cur + i * g
        h_next = o * self.activation(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        h, w = image_size
        return (torch.zeros(batch_size, self.out_channels, h, w, device=self.conv_i.weight.device),
                torch.zeros(batch_size, self.out_channels, h, w, device=self.conv_i.weight.device))
