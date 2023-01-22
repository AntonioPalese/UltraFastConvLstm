import torch
from model.backbone import resnet
import numpy as np



class conv_bn_relu(torch.nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,bias=False):
        super(conv_bn_relu,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels,out_channels, kernel_size,
            stride = stride, padding = padding, dilation = dilation,bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class parsingNet(torch.nn.Module):
    def __init__(self,cfg):
        super(parsingNet, self).__init__()

        self.w = cfg.model.size[1]
        self.h = cfg.model.size[0]
        self.cls_dim = tuple(cfg.model.cls_dim) 
        self.total_dim = np.prod(self.cls_dim)
        self.hidden_sizes = cfg.model.hidden_sizes
        self.num_layers = cfg.model.num_layers
        self.bias = cfg.model.bias
        self.seq_len = cfg.model.seq_len

        kernel_sizes = []
        for el in cfg.model.kernel_sizes:
            kernel_sizes.append((el,el))

        print('kernel_sizes :', kernel_sizes)

        self.kernel_sizes = kernel_sizes
        self.hidden_sizes = cfg.model.hidden_sizes
        #self.lstm_input_size = lstm_input_size
        backbone = cfg.model.backbone
        self.device = cfg.model.device

        # input : nchw,
        # output: (w+1) * sample_rows * 4
        self.model = resnet(backbone, pretrained=True)

        input_channel = 512 if backbone in ['34','18'] else 2048
        self.lstm = ConvLSTM(kernel_sizes=self.kernel_sizes, channel_input=input_channel, device=self.device, input_size=(9,25), bias=self.bias,
                             hidden_sizes=self.hidden_sizes, return_all_layers=False,num_layers=self.num_layers, seq_len=self.seq_len)

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(8*9*25, self.total_dim)
        )

        self.pool = torch.nn.Sequential(
            torch.nn.Conv2d(self.hidden_sizes[-1],256,1),
            torch.nn.Conv2d(256,8,1)
        )
    
        self.initialize_weights(self.cls)

    def forward(self, x):               

        unbinded = torch.unbind(x, dim=1)

        outs = []
        lasts_of_the_sequence = None
        for x in unbinded:
            x2, x3, fea = self.model(x)
            outs.append(fea)

        stacked_sequence = torch.stack(outs, dim=1)
 
        #print('shape after resnet : ', stacked_sequence.shape) #torch.Size([24, 3, 512, 9, 25])


        temporal_out, _ = self.lstm(stacked_sequence, None) #the initial state of the cells needs to be None

        test = temporal_out[:, -1, : ,: ,:]

        #print('shape after ConvLSTM, last image in sequence on the last Convlayer: ', test.shape)
        #torch.Size([24, 512, 9, 25])

        o=self.pool(test).view(-1,8 * 9 * 25)
        #print('shape after pooling layer an flattening ', o.shape) torch.Size([24, 1800])
        o=self.cls(o).view(-1,*self.cls_dim)
        #print('shape after classification layer and reviewing ', o.shape) torch.Size([24, 201, 18, 4])

        return o


    def initialize_weights(self, *models):
        for model in models:
            self.real_init_weights(model)

    def real_init_weights(self, m):
        if isinstance(m, list):
            for mini_m in m:
                self.real_init_weights(mini_m)
        else:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0.0, std=0.01)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Module):
                for mini_m in m.children():
                    self.real_init_weights(mini_m)
            else:
                raise ModuleNotFoundError()




class ConvLSTMnode(torch.nn.Module):
    def __init__(self, input_size, channel_input, hidden_size, kernel_size, bias, device):
        super(ConvLSTMnode,self).__init__()

        self.height, self.width = input_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bias = bias
        self.kernel_size = kernel_size
        self.channel_input = channel_input
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.device = device

        self.conv = torch.nn.Conv2d(in_channels=self.channel_input + hidden_size,
                                    kernel_size=kernel_size, out_channels=hidden_size*4,
                                    bias=self.bias,padding=self.padding)

    def forward(self, input_tensor, current_state):
        # input tensor (B, C, H, W)
        h_curr, c_curr = current_state

        combined = torch.cat([input_tensor, h_curr], dim=1)

        combined = self.conv(combined)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined, self.hidden_size, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_curr + i * g
        h_next = o*torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self,batch_size):
        h = torch.zeros((batch_size, self.hidden_size, self.height, self.width)).to(self.device)
        c = torch.zeros((batch_size, self.hidden_size, self.height, self.width)).to(self.device)

        return h, c

class ConvLSTM(torch.nn.Module):
    def __init__(self, kernel_sizes, hidden_sizes, input_size, num_layers, channel_input, bias, return_all_layers, device, seq_len):
        super(ConvLSTM, self).__init__()
        self.device = device
        self.channel_input = channel_input
        self.kernel_sizes = kernel_sizes
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_sizes = hidden_sizes
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.height, self.width = input_size
        self.seq_len = seq_len

        assert (isinstance(self.kernel_sizes, list) or isinstance(self.kernel_sizes, tuple)) and\
               (len(self.kernel_sizes) == self.num_layers)

        assert (isinstance(self.hidden_sizes, list) or isinstance(self.hidden_sizes, tuple)) and\
               (len(self.hidden_sizes) == self.num_layers)


        list_of_cells = []

        for i in range(self.num_layers):
            current_input_dimension = self.channel_input if i == 0 else self.hidden_sizes[i - 1]

            list_of_cells.append(ConvLSTMnode(input_size=self.input_size, device=self.device, bias=self.bias, channel_input=current_input_dimension,
                                              hidden_size=self.hidden_sizes[i], kernel_size=self.kernel_sizes[i]))

        self.list_of_cells = torch.nn.ModuleList(list_of_cells)


    def forward(self, input_tensor, hidden_state = None):


        self.batch_size = input_tensor.shape[0]

        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(self.batch_size)

        layer_output_list = []
        last_state_list = []


        cur_layer_input = input_tensor

        for i in range(self.num_layers):
            h, c = hidden_state[i]
            internal_output = []

            for t in range(self.seq_len):
                h, c = self.list_of_cells[i](input_tensor=cur_layer_input[:, t, :, :, :], current_state=(h, c))

                internal_output.append(h)

            layer_out = torch.stack(internal_output, dim=1)

            cur_layer_input = layer_out

            layer_output_list.append(layer_out)
            last_state_list.append((h, c))

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1]
            last_state_list = last_state_list[-1]

        #print(len(layer_output_list))
        #print(layer_output_list[0].shape)
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.list_of_cells[i].init_hidden(batch_size))

        return init_states



























