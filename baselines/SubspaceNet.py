#SubspaceNet implementation

import numpy as np
import scipy.io
d = scipy.io.loadmat('') #dataset path 
y_receive = d['y_receive'][:,:,2,:] #(16,100,5000)
target_azimuth = np.squeeze(d["target_azimuth"]).T  #(5000,4)

N, T, S = y_receive.shape
tau_max = 50
tau_max = min(tau_max, T - 1)
R_hat = np.zeros((S, N, N, tau_max + 1), dtype=np.complex64)

for tau in range(tau_max + 1):
    X0 = y_receive[:, :T - tau, :]
    X1 = y_receive[:, tau:, :]
    R_tau = np.einsum('nts,mts->nms', X0, np.conjugate(X1)) / (T - tau)
    R_hat[:, :, :, tau] = np.transpose(R_tau, (2, 0, 1))
R_ri = np.concatenate([R_hat.real, R_hat.imag], axis=1).astype(np.float32)

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import mat73

d = mat73.loadmat('/kaggle/input/train-set-all-snrs-all-snapshots/1_BS_4T_4U_5000_data_8.mat')
y_receive = d['y_receive'][:, :, 2, :]
target_azimuth = np.squeeze(d["target_azimuth"]).T

N, T, S = y_receive.shape
K = 4
tau_max = 50
tau_max = min(tau_max, T - 1)

R_hat = np.zeros((S, N, N, tau_max + 1), dtype=np.complex64)
for tau in range(tau_max + 1):
    X0 = y_receive[:, :T - tau, :]
    X1 = y_receive[:, tau:, :]
    R_tau = np.einsum('nts,mts->nms', X0, np.conjugate(X1)) / (T - tau)
    R_hat[:, :, :, tau] = np.transpose(R_tau, (2, 0, 1))
R_ri = np.concatenate([R_hat.real, R_hat.imag], axis=1).astype(np.float32)

gt = target_azimuth[:, :K].astype(np.float32)
if np.nanmax(np.abs(gt)) > np.pi:
    gt = np.deg2rad(gt).astype(np.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)

class SubspaceNetAutoEncoder(nn.Module):
    def __init__(self, N, tau_max):
        super().__init__()
        Cin = tau_max + 1
        self.enc1 = nn.Conv2d(Cin, 16, kernel_size=2, stride=1, padding=0, bias=True)
        self.enc2 = nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0, bias=True)
        self.enc3 = nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0, bias=True)
        self.dec1 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=1, padding=0, bias=True)
        self.dec2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=1, padding=0, bias=True)
        self.dec3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=0, bias=True)
        self.act = AReLU()

    def forward(self, R_ri):
        x = R_ri.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.enc1(x))
        x = self.act(self.enc2(x))
        x = self.act(self.enc3(x))
        x = self.act(self.dec1(x))
        x = self.act(self.dec2(x))
        x = self.dec3(x)
        return x.squeeze(1)

def post_process_covariance(ae_out, eps=1e-3):
    N = ae_out.shape[1] // 2
    Kre = ae_out[:, :N, :]
    Kim = ae_out[:, N:, :]
    Kc = torch.complex(Kre, Kim)
    I = torch.eye(N, device=ae_out.device, dtype=Kc.dtype).unsqueeze(0)
    R = Kc @ Kc.conj().transpose(-1, -2) + eps * I
    return R

def fb_average(R):
    N = R.shape[-1]
    J = torch.flip(torch.eye(N, device=R.device, dtype=R.dtype), dims=[1]).unsqueeze(0)
    return 0.5 * (R + J @ R.conj() @ J)

def diag_trace_offset(G, k):
    return torch.diagonal(G, offset=k, dim1=-2, dim2=-1).sum(dim=-1)

def poly_roots_companion(a):
    B, L = a.shape
    n = L - 1
    a0 = a[:, :1]
    a_norm = a / a0
    C = torch.zeros((B, n, n), dtype=a.dtype, device=a.device)
    C[:, 0, :] = -a_norm[:, 1:]
    C[:, 1:, :-1] = torch.eye(n - 1, dtype=a.dtype, device=a.device).unsqueeze(0)
    r = torch.linalg.eigvals(C)
    return r

def root_music_from_cov(R, K, d_over_lambda=0.5, forward_backward=True):
    if forward_backward:
        R = fb_average(R)
    evals, evecs = torch.linalg.eigh(R)
    idx = torch.argsort(evals, dim=-1, descending=True)
    evecs = torch.gather(evecs, -1, idx.unsqueeze(-2).expand_as(evecs))
    Un = evecs[:, :, K:]
    Gn = Un @ Un.conj().transpose(-1, -2)
    lags = list(range(-(N - 1), N))
    c = torch.stack([diag_trace_offset(Gn, lag) for lag in lags], dim=1)
    a = torch.flip(c, dims=[1])
    roots = poly_roots_companion(a)
    dist = torch.abs(torch.abs(roots) - 1.0)
    _, sel = torch.topk(dist, k=K, dim=1, largest=False)
    chosen = torch.gather(roots, 1, sel)
    phi = torch.atan2(chosen.imag, chosen.real)
    mu = phi / (2.0 * np.pi * d_over_lambda)
    mu = torch.clamp(mu.real, -1.0, 1.0)
    theta = torch.asin(mu)
    theta, _ = torch.sort(theta, dim=1)
    return theta

def periodic_diff(pred, gt, period=np.pi):
    return torch.remainder(pred - gt + period / 2.0, period) - period / 2.0

perms = torch.tensor(list(itertools.permutations(range(K))), device=device, dtype=torch.long)

def rmspe_perm_loss(pred, gt):
    B = pred.shape[0]
    predp = pred[:, perms]
    gtp = gt[:, None, :].expand(B, perms.shape[0], K)
    d = periodic_diff(predp, gtp, period=np.pi)
    mse = (d * d).mean(dim=-1)
    rmse = torch.sqrt(mse + 1e-12)
    return rmse.min(dim=1).values.mean()

X = torch.from_numpy(R_ri)
Y = torch.from_numpy(gt)
ds = TensorDataset(X, Y)

g = np.random.default_rng(0)
idx = g.permutation(S)
ntr = int(0.8 * S)
tr_idx = idx[:ntr]
te_idx = idx[ntr:]
tr_ds = torch.utils.data.Subset(ds, tr_idx.tolist())
te_ds = torch.utils.data.Subset(ds, te_idx.tolist())

batch_size = 64
tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=False)
te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, drop_last=False)

model = SubspaceNetAutoEncoder(N=N, tau_max=tau_max).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
eps_cov = 1e-3
epochs = 10

for epoch in range(epochs):
    model.train()
    for xb, yb in tr_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        ae = model(xb)
        R = post_process_covariance(ae, eps=eps_cov)
        pred = root_music_from_cov(R, K=K, d_over_lambda=0.5, forward_backward=True)
        loss = rmspe_perm_loss(pred, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

model.eval()
pred_all = []
gt_all = []
with torch.no_grad():
    for xb, yb in te_loader:
        xb = xb.to(device)
        ae = model(xb)
        R = post_process_covariance(ae, eps=eps_cov)
        pred = root_music_from_cov(R, K=K, d_over_lambda=0.5, forward_backward=True)
        pred_all.append(pred.cpu().numpy())
        gt_all.append(yb.numpy())

pred_all = np.concatenate(pred_all, axis=0)
gt_all = np.concatenate(gt_all, axis=0)

def avg_10log10_rmse_deg(est_rad, gt_rad, eps=1e-12):
    est = np.sort(est_rad, axis=1)
    gt  = np.sort(gt_rad,  axis=1)
    rmse_rad = np.sqrt(np.mean((est - gt)**2, axis=1))
    rmse_deg = np.degrees(rmse_rad)
    score_per_sample = 10.0 * np.log10(np.maximum(rmse_deg, eps))
    return score_per_sample.mean(), score_per_sample, rmse_deg

score_mean, score_per_sample, rmse_deg = avg_10log10_rmse_deg(pred_all, gt_all)
score_mean, rmse_deg.mean()
