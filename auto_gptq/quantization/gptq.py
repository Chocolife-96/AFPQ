import math
import os
import time
from logging import getLogger

import torch
import torch.nn as nn
import transformers

from .quantizer import Quantizer


logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer, format: str, gptq_quant: bool):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.pytorch_utils.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.format = format
        if format == 'nf':
            from .quantizer_nf4 import Quantizer_nf4
            self.quantizer = Quantizer_nf4()
        elif format == 'fp':
            from .quantizer_fp4 import Quantizer_fp4
            self.quantizer = Quantizer_fp4()
        else:
            self.quantizer = Quantizer()

        if gptq_quant == True:
            self.fasterquant = self.fasterquant
        else:
            self.fasterquant = self.fasterquant_rtn

    def add_batch(self, inp, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, group_size=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        g_idx = []
        scale = []
        scale2 = []
        zero = []
        now_idx = 1

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, group_size):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + group_size)], weight=True)
                if self.format == 'nf':
                    scale.append(quantizer.scale)
                    scale2.append(quantizer.scale2)
                elif self.format == 'fp':
                    scale.append(quantizer.scale)
                    scale2.append(quantizer.scale2)
                else:
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if group_size != -1:
                    if (i1 + i) % group_size == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + group_size)], weight=True)

                    if ((i1 + i) // group_size) - now_idx == -1:
                        if self.format == 'nf':
                            scale.append(self.quantizer.scale)
                            scale2.append(self.quantizer.scale2)
                        elif self.format == 'fp':
                            scale.append(self.quantizer.scale)
                            scale2.append(self.quantizer.scale2)
                        elif self.format == 'af':
                            scale.append(self.quantizer.scale)
                        else: # int
                            scale.append(self.quantizer.scale)
                            zero.append(self.quantizer.zero)
                        now_idx += 1

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0)) #
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:]) #

            if os.environ.get("DEBUG"):
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                logger.debug(torch.sum(Losses))

        torch.cuda.synchronize()
        logger.info(f'duration: {(time.time() - tick)}')
        logger.info(f'avg loss: {torch.sum(Losses).item() / self.nsamples}')

        group_size = group_size if group_size != -1 else self.columns
        if static_groups and actorder:
            g_idx = [perm[i] // group_size for i in range(self.columns)]
        else:
            g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if self.format == 'nf':
            if scale == []:
                scale.append(self.quantizer.scale)
            if scale2 == []:
                scale2.append(self.quantizer.scale2)
            scale = torch.cat(scale, dim=1)
            scale2 = torch.cat(scale2, dim=1)
            return scale, scale2, g_idx
        elif self.format == 'fp':
            if scale == []:
                scale.append(self.quantizer.scale)
            if scale2 == []:
                scale2.append(self.quantizer.scale2)
            scale = torch.cat(scale, dim=1)
            scale2 = torch.cat(scale2, dim=1)
            return scale, scale2, g_idx
        else: # int
            if scale == []:
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            return scale, zero, g_idx

    def fasterquant_rtn(
        self, blocksize=128, percdamp=.01, group_size=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        g_idx = []
        scale = []
        scale2 = []
        zero = []

        if group_size == -1:
            self.quantizer.find_params(W, weight=True)
            Q = self.quantizer.quantize(W)
        else:
            split_tensors = torch.split(W, group_size, dim=1)
            for i, split_tensor in enumerate(split_tensors):
                self.quantizer.find_params(split_tensor, weight=True)
                if self.format == 'nf':
                    scale.append(self.quantizer.scale)
                    scale2.append(self.quantizer.scale2)
                elif self.format == 'fp':
                    scale.append(self.quantizer.scale)
                    scale2.append(self.quantizer.scale2)
                else: # int
                    scale.append(self.quantizer.scale)
                    zero.append(self.quantizer.zero)
                Q[:, i*group_size:(i+1)*group_size] = self.quantizer.quantize(split_tensor)

        torch.cuda.synchronize()
        logger.info(f'duration: {(time.time() - tick)}')
        # logger.info(f'avg loss: {torch.sum(Losses).item()}')

        group_size = group_size if group_size != -1 else self.columns
        g_idx = [i // group_size for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(self.layer.weight.data)
        if os.environ.get("DEBUG"):
            logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        if self.format == 'nf':
            if scale == []:
                scale.append(self.quantizer.scale)
            if scale2 == []:
                scale2.append(self.quantizer.scale2)
            scale = torch.cat(scale, dim=1)
            scale2 = torch.cat(scale2, dim=1)
            return scale, scale2, g_idx
        elif self.format == 'fp':
            if scale == []:
                scale.append(self.quantizer.scale)
            if scale2 == []:
                scale2.append(self.quantizer.scale2)
            scale = torch.cat(scale, dim=1)
            scale2 = torch.cat(scale2, dim=1)
            return scale, scale2, g_idx
        else: # int
            if scale == []:
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            return scale, zero, g_idx

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
