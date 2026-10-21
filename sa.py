import torch
import math
from torch import nn
from torch.nn import functional as F

# 输入：x:[b,h,t,d]  position:[t]
def rope(x, position, base = 10000):
    # check
    if x.ndim != 4 or x.size(-1)%2 != 0:
        raise ValueError("x维度：[b,h,t,d],最后d要是双数")
    if position.ndim != 1 or position.size(0) != x.size(-2):
        raise ValueError("position长度要和x长度一样")
    d = x.size(-1)
    idx = torch.arange(0, d, 2, device=x.device)
    inv_freq = base**(-idx/d)
    angel = position[:,None] * inv_freq[None,:] # [t, d/2]
    cos = angel.cos()[None, None, :, :]
    sin = angel.sin()[None, None, :, :]
    enve = x[..., 0::2]
    odd = x[..., 1::2]
    enve = enve*cos - odd*sin
    odd = enve*sin - odd*cos
    out = torch.stack((enve, odd), dim=-1).flatten(-2)
    return out

def run():
    x = torch.randn(2,2,5,6)
    position = torch.arange(5)
    print(position.size(0), )
    x = rope(x, position)
    print(x.shape)

if __name__=='__main__':
    run()