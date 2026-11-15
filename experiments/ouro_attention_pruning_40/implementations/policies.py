"""Budget-aligned ports. Source formula fixtures live in test_policies.py.

All scoring sees only currently retained K and a recent-query buffer. Positions
remain absolute; no evicted K/V can be reconstructed from the auxiliary state.
"""
import math
import torch
import torch.nn.functional as F


def snap_scores(q, k, positions, n, window, kernel=5):
    # q/k [R,H,W_or_S,D]. Reevaluate the window on surviving keys, as SnapKV-D.
    q = q[..., -window:, :]
    logits = (q @ k.transpose(-1, -2)) / math.sqrt(k.shape[-1])
    qp = torch.arange(n-window,n,device=k.device)
    logits = logits.masked_fill(positions[..., None, :] > qp[:, None], float('-inf'))
    weights = logits.softmax(-1,dtype=torch.float32).to(q.dtype)
    score = weights[..., :-window].mean(-2)
    return F.avg_pool1d(score.flatten(0,1),kernel,stride=1,padding=kernel//2).reshape_as(score)


def redundancy(k):
    # Official R-KV: column mean after self/last-similar-column suppression.
    # Avoid materializing an N*N int64 index tensor (same integer decisions).
    z = k / (k.norm(dim=-1,keepdim=True)+1e-8)
    sim = z @ z.transpose(-1,-2)
    n=k.shape[-2];ix=torch.arange(n,device=k.device)
    sim[...,ix,ix]=0
    last=torch.where(sim>.5,ix.to(torch.int32),0).amax(-1).long()
    sim.scatter_(-1,last[...,None],0)
    return sim.mean(-2).softmax(-1)


def rkv_scores(q,k,window=8,kernel=7,mix_lambda=.1):
    logits=(q[...,-window:,:] @ k.transpose(-1,-2))/math.sqrt(k.shape[-1])
    importance=logits[...,:-window].softmax(-1,dtype=torch.float32).mean(-2).to(q.dtype)
    importance=F.max_pool1d(importance.flatten(0,1),kernel,stride=1,padding=kernel//2).reshape_as(importance)
    return mix_lambda*importance-(1-mix_lambda)*redundancy(k)[...,:-window]


def tri_scores(k, stats, omega, n):
    # Algebraic evaluation of the official mean-over-geometric-offsets formula.
    # K is already rotated at its original absolute position.
    k=k.float();d=k.shape[-1]//2;kr,ki=k[...,:d],k[...,d:]
    qr,qi,qa=stats['q_real'][...,None,:],stats['q_imag'][...,None,:],stats['q_abs'][...,None,:]
    offsets=2.**torch.arange(17,device=k.device,dtype=torch.float32)
    phase=(n-1+offsets[:,None])*omega[None,:]
    cs,ss=phase.cos().mean(0),phase.sin().mean(0)
    real=qr*kr+qi*ki;imag=qi*kr-qr*ki
    extra=(qa-torch.sqrt(qr*qr+qi*qi))*torch.sqrt(kr*kr+ki*ki)
    return (real*cs-imag*ss+extra).sum(-1)
