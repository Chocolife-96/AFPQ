import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import bitsandbytes as bnb
import torch.nn as nn
from functools import partial
import bitsandbytes.functional as bnbF


class SteNF4Quantizer(nn.Module):
    def __init__(self, bit, q_group_size=128):
        super().__init__()
        self.bit = bit
        self.q_group_size = q_group_size
    
    def forward(self, x):
        org_w_shape = x.shape

        # reshape to groupsize
        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            qx = x.reshape(-1, self.q_group_size)
        elif self.q_group_size == -1:
            qx = x.reshape(-1, x.shape[-1])
        assert qx.dim() == 2

        # Get the Max
        max_val = qx.amax(dim=1, keepdim=True)
        min_val = qx.amin(dim=1, keepdim=True)

        dev = qx.device
        scale = torch.maximum(torch.abs(max_val), torch.abs(min_val))

        dev = qx.device
        qx = qx / scale

        qx = self.round_pass(qx, dev)
        qx = qx * scale

        qx = qx.reshape(org_w_shape)
        return qx        

    def round_nf4(self, qx, dev):
        qx = torch.where(qx >= 0.8614784181118011,                                        torch.tensor(1.0).to(dev), qx)
        qx = torch.where((qx < 0.8614784181118011)    & (qx >= 0.6427869200706482),    torch.tensor(0.7229568362236023).to(dev), qx)
        qx = torch.where((qx < 0.6427869200706482)    & (qx >= 0.5016634166240692),    torch.tensor(0.5626170039176941).to(dev), qx)
        qx = torch.where((qx < 0.5016634166240692)    & (qx >= 0.3893125355243683),    torch.tensor(0.44070982933044434).to(dev), qx)
        qx = torch.where((qx < 0.3893125355243683)    & (qx >= 0.2920137718319893),    torch.tensor(0.33791524171829224).to(dev), qx)
        qx = torch.where((qx < 0.2920137718319893)    & (qx >= 0.2035212516784668),    torch.tensor(0.24611230194568634).to(dev), qx)
        qx = torch.where((qx < 0.2035212516784668)    & (qx >= 0.1202552504837513),    torch.tensor(0.16093020141124725).to(dev), qx)
        qx = torch.where((qx < 0.1202552504837513)    & (qx >= 0.03979014977812767),   torch.tensor(0.07958029955625534).to(dev), qx)
        qx = torch.where((qx < 0.03979014977812767)   & (qx >= -0.045525018125772476),     torch.tensor(0).to(dev), qx)

        # qx = torch.where(qx >= -0.045525018125772476,                                     torch.tensor(0).to(dev), qx)
        qx = torch.where((qx < -0.045525018125772476) & (qx >= -0.13791173323988914),  torch.tensor(-0.09105003625154495).to(dev), qx)
        qx = torch.where((qx < -0.13791173323988914)  & (qx >= -0.23460740596055984),  torch.tensor(-0.18477343022823334).to(dev), qx)
        qx = torch.where((qx < -0.23460740596055984)  & (qx >= -0.33967943489551544),  torch.tensor(-0.28444138169288635).to(dev), qx)
        qx = torch.where((qx < -0.33967943489551544)  & (qx >= -0.4599952697753906),   torch.tensor(-0.39491748809814453).to(dev), qx)
        qx = torch.where((qx < -0.4599952697753906)   & (qx >= -0.6106329262256622),   torch.tensor(-0.5250730514526367).to(dev), qx)
        qx = torch.where((qx < -0.6106329262256622)   & (qx >= -0.8480964004993439),   torch.tensor(-0.6961928009986877).to(dev), qx)
        qx = torch.where(qx < -0.8480964004993439,                                        torch.tensor(-1.0).to(dev), qx)

        return qx

    def round_pass(self, qx, dev):
        y = qx
        y_nf4 = self.round_nf4(y, dev)
 
        return (y_nf4 - y).detach() + y

class SteN2F4Quantizer(nn.Module):
    def __init__(self, bit, q_group_size=128):
        super().__init__()
        self.bit = bit
        self.q_group_size = q_group_size
    
    def forward(self, x):
        org_w_shape = x.shape

        # reshape to groupsize
        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            qx = x.reshape(-1, self.q_group_size)
        elif self.q_group_size == -1:
            qx = x.reshape(-1, x.shape[-1])
        assert qx.dim() == 2

        # Get the Min Max
        max_val = qx.amax(dim=1, keepdim=True)
        min_val = qx.amin(dim=1, keepdim=True)

        scale_pos = torch.abs(max_val)
        scale_neg = torch.abs(min_val)

        dev = qx.device
        x_pos = torch.zeros_like(qx)
        x_neg = torch.zeros_like(qx)
        x_pos = torch.where(qx >= 0, qx, x_pos)
        x_neg = torch.where(qx < 0, qx, x_neg)
        q_pos = x_pos / scale_pos
        q_neg = x_neg / scale_neg

        q_pos, q_neg = self.round_pass(q_pos, q_neg, dev)

        qx = q_pos * scale_pos + q_neg * scale_neg

        qx = qx.reshape(org_w_shape)
        return qx        

    def round_nf4(self, q_pos, q_neg, dev):
        q_pos = torch.where(q_pos >= 0.8614784181118011,                                        torch.tensor(1.0).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.8614784181118011)    & (q_pos >= 0.6427869200706482),    torch.tensor(0.7229568362236023).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.6427869200706482)    & (q_pos >= 0.5016634166240692),    torch.tensor(0.5626170039176941).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.5016634166240692)    & (q_pos >= 0.3893125355243683),    torch.tensor(0.44070982933044434).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.3893125355243683)    & (q_pos >= 0.2920137718319893),    torch.tensor(0.33791524171829224).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.2920137718319893)    & (q_pos >= 0.2035212516784668),    torch.tensor(0.24611230194568634).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.2035212516784668)    & (q_pos >= 0.1202552504837513),    torch.tensor(0.16093020141124725).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.1202552504837513)    & (q_pos >= 0.03979014977812767),   torch.tensor(0.07958029955625534).to(dev), q_pos)
        q_pos = torch.where(q_pos < 0.03979014977812767,                                        torch.tensor(0).to(dev), q_pos)

        q_neg = torch.where(q_neg >= -0.045525018125772476,                                     torch.tensor(0).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.045525018125772476) & (q_neg >= -0.13791173323988914),  torch.tensor(-0.09105003625154495).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.13791173323988914)  & (q_neg >= -0.23460740596055984),  torch.tensor(-0.18477343022823334).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.23460740596055984)  & (q_neg >= -0.33967943489551544),  torch.tensor(-0.28444138169288635).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.33967943489551544)  & (q_neg >= -0.4599952697753906),   torch.tensor(-0.39491748809814453).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.4599952697753906)   & (q_neg >= -0.6106329262256622),   torch.tensor(-0.5250730514526367).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.6106329262256622)   & (q_neg >= -0.8480964004993439),   torch.tensor(-0.6961928009986877).to(dev), q_neg)
        q_neg = torch.where(q_neg < -0.8480964004993439,                                        torch.tensor(-1.0).to(dev), q_neg)

        return q_pos, q_neg


    def round_pass(self, q_pos, q_neg, dev):
        y_grad_pos, y_grad_neg = q_pos, q_neg
        y_pos, y_neg = self.round_nf4(q_pos, q_neg, dev)
        
        return (y_pos - y_grad_pos).detach() + y_grad_pos, (y_neg - y_grad_neg).detach() + y_grad_neg

class SteNF3Quantizer(nn.Module):
    def __init__(self, bit, q_group_size=128):
        super().__init__()
        self.bit = bit
        self.q_group_size = q_group_size
    
    def forward(self, x):
        org_w_shape = x.shape

        # reshape to groupsize
        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            qx = x.reshape(-1, self.q_group_size)
        elif self.q_group_size == -1:
            qx = x.reshape(-1, x.shape[-1])
        assert qx.dim() == 2

        # Get the Min Max
        max_val = qx.amax(dim=1, keepdim=True)
        min_val = qx.amin(dim=1, keepdim=True)

        dev = qx.device
        scale = torch.maximum(torch.abs(max_val), torch.abs(min_val))

        qx = qx / scale

        qx = self.round_pass(qx, dev)

        qx = qx * scale

        qx = qx.reshape(org_w_shape)

        return qx
    
    def round_pass(self, qx, dev):
        y_grad = qx
        qy = self.round_nf3(qx, dev)
        
        return (qy - y_grad).detach() + y_grad
    
    def round_nf3(self, q, dev):
        q = torch.where(q >= 0.8114928305149078,                                    torch.tensor(1.0).to(dev), q)
        q = torch.where((q < 0.8114928305149078)    & (q >= 0.5024898052215576),    torch.tensor(0.6229856610298157).to(dev), q)
        q = torch.where((q < 0.5024898052215576)    & (q >= 0.2826657369732857),    torch.tensor(0.3819939494132996).to(dev), q)
        q = torch.where((q < 0.2826657369732857)    & (q >= 0.0916687622666359),   torch.tensor(0.1833375245332718).to(dev), q)
        q = torch.where((q < 0.0916687622666359)   & (q >= -0.1234657019376755), torch.tensor(0).to(dev), q)
        q = torch.where((q < -0.1234657019376755) & (q >= -0.39097706973552704),  torch.tensor(-0.2469314038753510).to(dev), q)
        q = torch.where((q < -0.39097706973552704)  & (q >= -0.7675113677978516),  torch.tensor(-0.5350227355957031).to(dev), q)
        q = torch.where(q < -0.7675113677978516,                                    torch.tensor(-1.0).to(dev), q)
        return q


class SteN2F3Quantizer(nn.Module):
    def __init__(self, bit, q_group_size=128):
        super().__init__()
        self.bit = bit
        self.q_group_size = q_group_size
    
    def forward(self, x):
        org_w_shape = x.shape

        # reshape to groupsize
        if self.q_group_size > 0:
            assert org_w_shape[-1] % self.q_group_size == 0
            qx = x.reshape(-1, self.q_group_size)
        elif self.q_group_size == -1:
            qx = x.reshape(-1, x.shape[-1])
        assert qx.dim() == 2

        # Get the Min Max
        max_val = qx.amax(dim=1, keepdim=True)
        min_val = qx.amin(dim=1, keepdim=True)

        
        scale_pos = torch.abs(max_val)
        scale_neg = torch.abs(min_val)

        dev = qx.device
        x_pos = torch.zeros_like(qx)
        x_neg = torch.zeros_like(qx)
        x_pos = torch.where(qx >= 0, qx, x_pos)
        x_neg = torch.where(qx < 0, qx, x_neg)
        q_pos = x_pos / scale_pos
        q_neg = x_neg / scale_neg

        q_pos, q_neg = self.round_pass(q_pos, q_neg, dev)

        qx = q_pos * scale_pos + q_neg * scale_neg

        qx = qx.reshape(org_w_shape)

        return qx
    
    def round_n2f3(self, q_pos, q_neg, dev):
        q_pos = torch.where(q_pos >= 0.8114928305149078,                                        torch.tensor(1.0).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.8114928305149078)    & (q_pos >= 0.5024898052215576),    torch.tensor(0.6229856610298157).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.5024898052215576)    & (q_pos >= 0.2826657369732857),    torch.tensor(0.3819939494132996).to(dev), q_pos)
        q_pos = torch.where((q_pos < 0.2826657369732857)    & (q_pos >= 0.0916687622666359),    torch.tensor(0.1833375245332718).to(dev), q_pos)
        q_pos = torch.where(q_pos < 0.0916687622666359,                                        torch.tensor(0).to(dev), q_pos)

        q_neg = torch.where(q_neg >= -0.1234657019376755,                                     torch.tensor(0).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.1234657019376755)   & (q_neg >= -0.39097706973552704),   torch.tensor(-0.2469314038753510).to(dev), q_neg)
        q_neg = torch.where((q_neg < -0.39097706973552704)   & (q_neg >= -0.7675113677978516),   torch.tensor(-0.5350227355957031).to(dev), q_neg)
        q_neg = torch.where(q_neg < -0.7675113677978516,                                        torch.tensor(-1.0).to(dev), q_neg)

        return q_pos, q_neg

    def round_pass(self, q_pos, q_neg, dev):
        y_grad_pos, y_grad_neg = q_pos, q_neg
        y_pos, y_neg = self.round_n2f3(q_pos, q_neg, dev)
        
        return (y_pos - y_grad_pos).detach() + y_grad_pos, (y_neg - y_grad_neg).detach() + y_grad_neg



