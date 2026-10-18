import torch
import math
from torch import nn
from torch.nn import functional as F

    
class self_attention(nn.Module):
    def __init__(self, d_model, D_k=None, D_v=None):
        super().__init__()
        D_k = d_model if D_k == None else D_k
        D_v = d_model if D_v == None else D_v
        self.q_proj = nn.Linear(d_model, D_k)
        self.k_proj = nn.Linear(d_model, D_k)
        self.v_proj = nn.Linear(d_model, D_v)
        
    def forward(self, x, mask=False):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        score = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        if mask:
            t = Q.size(-2)
            blocked = torch.ones(t, t, device=Q.device, dtype=bool).triu(1)
            score = score.masked_fill(blocked, float('-inf'))

        weight = torch.softmax(score, dim=-1)

        return weight @ V , weight

def run():
    x = torch.randn(1, 4, 5, dtype=torch.float32, requires_grad=True)
    model = self_attention(5, 6, 2)
    out, _ = model(x, mask=True)
    print(out.shape)
    assert out.shape == (1, 4, 2)


if __name__ == '__main__':
    run()



