import torch.nn as nn
from torch.nn import init
import torch
from torch.autograd import Variable

class GRU_concat(nn.Module):
    def __init__(self, ninp, nhid):
        super(GRU_concat, self).__init__()
        self.in_dim = ninp
        self.mem_dim = nhid

        self.ioux = nn.Linear(2 * self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)

        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, sememe_h, hx):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []

        for time in range(max_time):
            new_input = torch.cat([inputs[time], sememe_h[time]], dim = 1)
            next_hx = self.node_forward(new_input, hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack(output, 0), hx

class GRU_base(nn.Module):
    def __init__(self, ninp, nhid):
        super(GRU_base, self).__init__()
        self.in_dim = ninp
        self.mem_dim = nhid

        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.Uh]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, sememe_h, hx):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack(output, 0), hx

class GRU_gate(nn.Module):
    def __init__(self, ninp, nhid):
        super(GRU_gate, self).__init__()
        self.in_dim = ninp
        self.mem_dim = nhid

        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.in_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fh_s, self.fs, self.Uh, self.W_c]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx
        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)
        o_c = self.fx_s(inputs) + self.fh_s(child_h) + self.fs(sememe_h)
        o_c = torch.sigmoid(o_c)
        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta) + torch.mul(o_c, torch.tanh(self.W_c(sememe_h)))
        return h

    def forward(self, inputs, sememe_h, hx):
        max_time, batch_size, _ = inputs.size()
        output = []

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack(output, 0), hx

class GRU_cell(nn.Module):
    def __init__(self, ninp, nhid):
        super(GRU_cell, self).__init__()
        self.in_dim = ninp
        self.mem_dim = nhid

        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.Uh_s = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.Uh, self.Uh_s]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_h, hx):
        child_h = hx

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h)) + self.Uh_s(torch.mul(r, sememe_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs, sememe_h, hx):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_h[time], hx)
            output.append(next_hx)
            hx = next_hx

        return torch.stack(output, 0), hx


class LSTM_base(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(LSTM_base, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fh, self.fs]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_c = hx[0]
        child_h = hx[1]
        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, hidden):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = hidden
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack(output, 0), hx

class LSTM_cell(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(LSTM_cell, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fh, self.fs]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_c, sememe_h, hx):
        child_c = hx[0]
        child_h = hx[1]
        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        f_s = torch.sigmoid(
            self.fs(sememe_h) + self.fx_s(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        fc_s = torch.mul(f_s, sememe_c) # part of memory cell induced by sememe-child
        c = torch.mul(i, u) + fc + fc_s #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, sememe_c, sememe_h, hidden):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = hidden
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], sememe_c[time], sememe_h[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack(output, 0), hx

class LSTM_gate(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(LSTM_gate, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 4 * self.mem_dim)
        #ious是专门处理sememe传过来的c 和 h，c和h都是mem_dim维的
        self.ious = nn.Linear(self.in_dim, 4 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        #self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.W_c = nn.Linear(self.in_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fh, self.W_c]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, emb_s, hx):
        child_c = hx[0]
        child_h = hx[1]

        iou = self.ioux(inputs) + self.iouh(child_h) + self.ious(emb_s)
        f, i, o, o_c = torch.split(iou, iou.size(1) // 4, dim=1)
        f, i, o, o_c = torch.sigmoid(f), torch.sigmoid(i), torch.sigmoid(o), torch.sigmoid(o_c)
        c_telta = self.fx(inputs) + self.fh(child_h)
        c_telta = torch.tanh(c_telta)
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, c_telta) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c)) + torch.mul(o_c, torch.tanh(self.W_c(emb_s)))
        return (c, h)

    def forward(self, inputs, emb_s, hidden):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = hidden
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], emb_s[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack(output, 0), hx

class LSTM_concat(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(LSTM_concat, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.ioux = nn.Linear(2 * self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.ious = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(2 * self.in_dim, self.mem_dim)
        self.fx_s = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fs = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.ious, self.fx, self.fx_s, self.fh, self.fs]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs,  hx):
        child_c = hx[0]
        child_h = hx[1]
        iou = self.ioux(inputs) + self.iouh(child_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(child_h) + self.fx(inputs)
        )
        fc = torch.mul(f, child_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc
        h = torch.mul(o, torch.tanh(c))
        return (c, h)

    def forward(self, inputs, hidden):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = hidden
        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx[1])
            hx = next_hx
        return torch.stack(output, 0), hx

class SememeSumLstm(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.fh]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, sememe_c, sememe_h):
        iou = self.ioux(inputs) + self.iouh(sememe_h)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        f = torch.sigmoid(
            self.fh(sememe_h) + self.fx(inputs)
        )
        fc = torch.mul(f, sememe_c) #part of memory cell induced by word-child
        c = torch.mul(i, u) + fc #sum means sigma
        h = torch.mul(o, torch.tanh(c))
        return c, h

    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        c = []
        h = []
        new_c, new_h = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_(), inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()
        for time in range(max_time):
            new_c, new_h = self.node_forward(inputs[time], new_c, new_h)
            c.append(new_c)
            h.append(new_h)
        return torch.stack(c, 0), torch.stack(h, 0)

class SememeSumLstm_GRU(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm_GRU, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 2 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.Uh = nn.Linear(self.mem_dim, self.mem_dim)
        self.reset_parameters()

    def reset_parameters(self):
        layers = [self.ioux, self.iouh, self.fx, self.Uh]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

    def node_forward(self, inputs, hx):
        child_h = hx
        iou = self.ioux(inputs) + self.iouh(child_h)
        z, r = torch.split(iou, iou.size(1) // 2, dim=1)
        z, r = torch.sigmoid(z), torch.sigmoid(r)

        h_telta = self.fx(inputs) + self.Uh(torch.mul(r, child_h))
        h_telta = torch.tanh(h_telta)
        h = torch.mul((1-z), child_h) + torch.mul(z, h_telta)
        return h

    def forward(self, inputs):
        # hx: (child_c, child_h)
        max_time, batch_size, _ = inputs.size()
        output = []
        hx = inputs[0][0].detach().new(batch_size, self.mem_dim).fill_(0.).requires_grad_()

        for time in range(max_time):
            next_hx = self.node_forward(inputs[time], hx)
            output.append(next_hx)
            hx = next_hx
        return torch.stack(output, 0)

class SememeSumLstm_old(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm_old, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.reset_parameters()
    def node_forward(self, inputs):
        iou = self.ioux(inputs)# three Wx+b
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        c = torch.mul(i, u)
        h = torch.mul(o, torch.tanh(c))
        return c, h
    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        c = []
        h = []
        for time in range(max_time):
            new_c, new_h = self.node_forward(inputs[time])
            c.append(new_c)
            h.append(new_h)
        return torch.stack(c, 0), torch.stack(h, 0)

    def reset_parameters(self):
        layers = [self.ioux]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)

class SememeSumLstm_GRU_old(nn.Module):
    def __init__(self, sememe_dim, mem_dim):
        super(SememeSumLstm_GRU_old, self).__init__()
        self.in_dim = sememe_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 2 * self.mem_dim)
        self.reset_parameters()
    def node_forward(self, inputs):
        iou = self.ioux(inputs)# three Wx+b
        i, o = torch.split(iou, iou.size(1) // 2, dim=1)
        i, o = torch.sigmoid(i), torch.tanh(o)

        h = torch.mul(i,o)
        return h
    def forward(self, inputs):
        max_time, batch_size, _ = inputs.size()
        h = []
        for time in range(max_time):
            new_h = self.node_forward(inputs[time])
            h.append(new_h)
        return torch.stack(h, 0)

    def reset_parameters(self):
        layers = [self.ioux]
        for layer in layers:
            init.kaiming_normal_(layer.weight)
            if layer.bias is not None:
                init.constant_(layer.bias, val=0)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, sememe_dim, sememe_size, model_type, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.model_type = model_type
        self.sememe_dim = sememe_dim
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.emb_sememe = nn.Embedding(sememe_size, sememe_dim)
        if 'LSTM' in self.model_type:
            self.sememesumlstm = SememeSumLstm_old(sememe_dim, nhid)
        elif 'GRU' in self.model_type:
            self.sememesumlstm = SememeSumLstm_GRU_old(sememe_dim, nhid)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers-1, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers-1, nonlinearity=nonlinearity, dropout=dropout)

        self.LSTM = eval(model_type)(ninp, nhid)
        self.decoder = nn.Linear(nhid, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, input_s, hidden):
        #sememe propagate
        emb = self.drop(self.encoder(input))
        new_input = None
        emb_sememe = self.drop(self.emb_sememe.weight)
        input_sememe = []
        for i in range(input_s.size()[0]):
            input_sememe.append(torch.mm(input_s[i].float(), emb_sememe))
        input_sememe = torch.stack(input_sememe, dim = 0)

        if self.model_type == 'LSTM_gate':
            new_input, hidden_lstm = self.LSTM(emb, input_sememe, hidden[0])
        elif self.model_type == 'LSTM_concat':
            emb = torch.cat([emb, input_sememe], dim = 2)
            new_input, hidden_lstm = self.LSTM(emb,  hidden[0])
        elif self.model_type == 'LSTM_cell':
            sememe_c, sememe_h = self.sememesumlstm(input_sememe)
            #sememe_c = self.drop(sememe_c)
            #sememe_h = self.drop(sememe_h)
            new_input, hidden_lstm = self.LSTM(emb, sememe_c, sememe_h, hidden[0])
        elif self.model_type == 'LSTM_base':
            new_input, hidden_lstm = self.LSTM(emb, hidden[0])
        elif self.model_type == 'GRU_cell':
            sememe_h = self.sememesumlstm(input_sememe)
            #sememe_h = self.drop(sememe_h)
            new_input, hidden_lstm = self.LSTM(emb, sememe_h, hidden[0])
        elif self.model_type == 'GRU_gate':
            new_input, hidden_lstm = self.LSTM(emb, input_sememe, hidden[0])
        elif self.model_type == 'GRU_base':
            sememe_h = self.sememesumlstm(input_sememe)
            new_input, hidden_lstm = self.LSTM(emb, sememe_h, hidden[0])
        elif self.model_type == 'GRU_concat':
            new_input, hidden_lstm = self.LSTM(emb, input_sememe, hidden[0])
        else:
            raise NameError

        new_input = self.drop(new_input)
        output, hidden_rnn = self.rnn(new_input, hidden[1])
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        hidden = (hidden_lstm, hidden_rnn)
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        if 'LSTM' in self.model_type:
            weight = next(self.parameters())
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    (weight.new_zeros(self.nlayers-1, bsz, self.nhid),
                    weight.new_zeros(self.nlayers-1, bsz, self.nhid)))
        elif 'GRU' in self.model_type:
            weight = next(self.parameters())
            return (weight.new_zeros(bsz, self.nhid),
                    weight.new_zeros(self.nlayers-1, bsz, self.nhid))
        else:
            raise NameError
