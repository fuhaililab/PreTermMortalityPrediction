import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import copy


def clones( module, N):
    """Clone the module
    Args:
        module(nn.Module): The torch module you want to clone
        N(int): Time you want to clone to.

    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(N))




class LSTMLayer(nn.Module):
    """General propose LSTM layer class
    Args:
        input_size(int): input dimension
        hidden_size(int): hidden(output) dimension
        num_layers(int): number of LSTM layer
        drop_prob(float): dropout probability.

    """
    def __init__(self,input_size,hidden_size,num_layers,drop_prob=.0):
        super(LSTMLayer, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=False,
                          batch_first=True)
        self.dropout=nn.Dropout(drop_prob)

    def forward(self,x,lengths):
        orig_len=x.size(1)
        lengths,sort_idx=lengths.sort(0,descending=True)
        x=x[sort_idx]
        #pack sequence for lstm input
        x=pack_padded_sequence(x,lengths,batch_first=True)
       # self.lstm.flatten_parameters()
        x,_=self.lstm(x)
        # pad back from packed sequence
        x, _=pad_packed_sequence(x,batch_first=True,total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]
        x=self.dropout(x)

        return(x)



class HighwayLayer(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayLayer, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x



class GlobalProjectLayer(nn.Module):
    """Global project layer for fixed variable learning
    Args:
        input_size(int): input dimension
        hidden_size(int): hidden(output) dimension
        seq_len(int): the time length of vital sign data
        drop_prob(float): dropout probability

    """
    def __init__(self,input_size,hidden_size,seq_len,drop_prob):
        super(GlobalProjectLayer, self).__init__()
        self.proj=nn.Linear(input_size,hidden_size)
        self.dropout=nn.Dropout(drop_prob)
        self.seq_len=seq_len

    def forward(self,x):
        x=self.proj(x)
        x=self.dropout(x)
        x=x.unsqueeze(1)
        x=x.repeat((1,self.seq_len,1))
        return F.relu(x)

class TimeWiseGateBlock(nn.Module):
    """Time-wise gate block to incorporate vital sign feature and fixed variable feature
    Args:
        hidden_size(int): the hidden dimension of vital sign feature and fixed variable feature

    """
    def __init__(self,hidden_size):
        super(TimeWiseGateBlock, self).__init__()
        self.proj=nn.Linear(4*hidden_size,1)
    def forward(self,x,g):
        m=torch.cat([x,g,torch.mul(x,g),torch.sub(x,g)],dim=-1)
        gate=F.sigmoid(self.proj(m))
        return torch.mul(gate,x)+torch.mul((1-gate),g)

class PointWiseGateBlock(nn.Module):
    """Point-wise gate block to incorporate vital sign feature and fixed variable feature
        Args:
        hidden_size(int): the hidden dimension of vital sign feature and fixed variable feature

    """
    def __init__(self,hidden_size):
        super(PointWiseGateBlock, self).__init__()
        self.proj=nn.Linear(4*hidden_size,hidden_size)
    def forward(self,x,g):
        m=torch.cat([x,g,torch.mul(x,g),torch.sub(x,g)],dim=-1)
        gate=torch.sigmoid(self.proj(m))
        return torch.mul(gate,x)+torch.mul((1-gate),g)


class ResidualLayerConnection(nn.Module):
    """Residual layer connection block for sequence data
    Args:
        hidden_size(int): hidden dimension
        layer(nn.Module): The residual component
        N(int): Number of residual component
    """
    def __init__(self,hidden_size,layer,N):
        super(ResidualLayerConnection, self).__init__()
        self.layers=clones(layer,N)
        self.hidden_size=hidden_size
        self.task=nn.Linear(N*hidden_size,N)
        self.N=N

    def forward(self,x,lengths):
        l_list=[x]
        for l in self.layers:
            out=l(l_list[-1],lengths)
            l_list.append(out)

        L=self.task(torch.cat(l_list[1:],dim=-1))
        L=L.transpose(-1,-2)
        attn=F.softmax(L,dim=1)
        attn=attn.unsqueeze(-1)
        l_total=torch.cat(l_list[1:],dim=1)
        batch_size=x.size(0)
        l_total=l_total.view(batch_size,self.N,-1,self.hidden_size)
        sum=torch.mul(l_total,attn).sum(dim=1)
        return sum



class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable convolution
    Args:
        in_channels: number of channels in the input, here is the hidden size of encoder
        out_channels:number of channels in the output, here is also the hidden size of encoder
    """
    def __init__(self, in_channels,out_channels,kernel_size,padding=0, bias=True):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.conv1=nn.Conv1d(in_channels=in_channels,
                             out_channels=in_channels,
                             kernel_size=kernel_size,
                             groups=in_channels,
                             padding=0,
                             bias=False)
        self.conv2=nn.Conv1d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=1,
                             groups=1,
                             padding=padding,
                             bias=bias)
        self.bn=nn.BatchNorm1d(out_channels)
        self.kernel_size=kernel_size
        self.hidden_size=in_channels

    def forward(self,x):
        # padding in the beginning of sequence, thus cannot get information from later time steps
        batch_size=x.size(0)
        padding=torch.zeros((batch_size,self.kernel_size-1,self.hidden_size))
        padding=padding.to(x.device)
        x=torch.cat([padding,x],dim=1)
        x = x.transpose(-1, -2).contiguous()
        x = F.relu(self.bn(self.conv2(self.conv1(x))))
        return x.transpose(-1,-2).contiguous()

class Detector(nn.Module):
    """Detector Block
    Args:
        input_channel(int): input dimension
        kernel_size(int): kernel size
        drop_prob(float):dropout probability
        N(int): number of CNN layer

    """
    def __init__(self,input_channel,kernel_size,drop_prob,N):
        super(Detector, self).__init__()
        cnn=DepthwiseSeparableConv1d(input_channel,input_channel,kernel_size)
        self.cnns=clones(copy.deepcopy(cnn),N)
        self.dropout=nn.Dropout(drop_prob)
        self.proj=nn.Linear(input_channel,1)

    def forward(self,x):
        for l in self.cnns:
            x=l(x)
        x=self.dropout(x)
        x=self.proj(x)
        batch_size=x.size(0)

        return x.view(batch_size,-1)

class Verification(nn.Module):
    """Verification block
    Args:
        hidden_size(int): hidden(output) dimension
    """
    def __init__(self,hidden_size):
        super(Verification, self).__init__()
        self.final_proj_noworry = nn.Linear(hidden_size, 2)
        self.final_proj_worry=nn.Linear(hidden_size,2)
        self.p1=LSTMLayer(2,2,1,0)
        self.p2=LSTMLayer(2,1,1,0)

    def forward(self,final_x,worry_time,lengths):
        noworry_det = self.final_proj_noworry(final_x)
        worry_det = self.final_proj_worry(final_x)
        noworry_det[worry_time] = worry_det[worry_time]

        p1=self.p1(noworry_det,lengths)
        p2=self.p2(worry_det,lengths)
        adjust_det=noworry_det-p1+p2
        return noworry_det,adjust_det

class DeepPBSMonitor(nn.Module):
    """DeepPBSMonitor Model
    Args:
        input_size(int):input dimension for vital sign data
        global_input_size(int):input dimension for fixed variable
        seq_len(int): time sequence length
        hidden_size(int):hidden dimension
        num_highway_layer(int): number of highway layer
        num_cnn(int): number of cnn layer in detector block
        drop_prob(float):dropout probability
    Return:
        noworry_det: prediction result without verification
        adjust_det: prediction result after verification
        det: detection probability result
        turning_point: final detected turning point
    """

    def __init__(self,input_size,global_input_size,seq_len,hidden_size,num_highway_layer,num_cnn,drop_prob):
        super(DeepPBSMonitor, self).__init__()
        self.seq_len=seq_len
        self.inital_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.Dropout(drop_prob), nn.ReLU())
        lstm = LSTMLayer(hidden_size, hidden_size, 1, drop_prob)
        self.residual_lstm = ResidualLayerConnection(hidden_size, lstm, 3)
        self.highway = nn.Sequential(HighwayLayer(num_highway_layer, hidden_size), nn.Dropout(drop_prob))

        self.final_lstm = LSTMLayer(hidden_size, hidden_size, 1, drop_prob)
        self.gate = PointWiseGateBlock(hidden_size=hidden_size)
        self.detector=Detector(hidden_size,7,drop_prob,num_cnn)
        self.verifier=Verification(hidden_size)
        self.global_proj = GlobalProjectLayer(global_input_size, hidden_size, seq_len, drop_prob)

    def forward(self, x, global_x, lengths):
        x = self.inital_proj(x)
        x = self.highway(x)
        x = self.residual_lstm(x, lengths)
        global_x = self.global_proj(global_x)
        final_x = self.gate(x, global_x)
        final_x = self.final_lstm(final_x, lengths)

        #detection
        det=self.detector(final_x)
        #get turning point
        mask=torch.zeros_like(det)
        for i in range(mask.size(0)):
            mask[i,:lengths[i]]=1
        mask = mask.type(torch.float32)
        det = mask * det + (1 - mask) * -1e30
        turning_point=F.softmax(det,dim=-1)
        if self.training:
            turning_point=torch.argmax(turning_point,dim=-1)
        else:
            #You can design your own evaluation method
            eva_turning_point=torch.argmax(turning_point,dim=-1)
            turning_point=eva_turning_point

        #verification
        worry_time=torch.ones_like(det)
        for i in range(worry_time.size(0)):
            worry_time[i,:turning_point[i]+1]=0
        worry_time=worry_time.bool()
        noworry_det,adjust_det=self.verifier(final_x,worry_time,lengths)

        return noworry_det.view(-1, 2),adjust_det.view(-1,2),det,turning_point



class FocalLoss(nn.Module):

    def __init__(self,weight,
                 gamma=2):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets,lengths):
        weights = torch.ones(inputs.size(0), device=inputs.get_device())
        for i in range(lengths.size(0)):
            if targets[i]==lengths[i]-1:
                weights[i] =self.weight
        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt)  # prevents nans when probability 0
        nll_loss = F.nll_loss((1-pt)**self.gamma*log_pt, targets,reduction="none")
        nll_loss=nll_loss*weights
        return nll_loss.mean()
