import os, math, time, atexit, signal, json, tempfile
import numpy as np
import scipy.io
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


@dataclass
class Config:
    data_path: str = "" #enter paths
    out_dir: str = ""
    batch_size: int = 256
    num_workers: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    betas_T: int = 10 # diffusion steps
    beta_schedule: str = "cosine"
    hidden: int = 512
    layers: int = 4
    time_emb_dim: int = 128
    cond_emb_dim: int = 256
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples_per_test: int = 8
    ddim: bool = False
    ddim_eta: float = 0
    test_size: float = 0.2
    db_eps: float = 1e-12
    save_every: int = 5
    use_amp: bool = True 

cfg = Config()
os.makedirs(cfg.out_dir, exist_ok=True)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
set_seed(cfg.seed)

# Load data
with np.load(cfg.data_path, allow_pickle=False) as data:
    y_receive = data['y_receive']           # (N,16) complex, N samples
    target_azimuth = data['target_azimuth'] # (N,4) radians

#print(y_receive.shape)
#print(target_azimuth.shape)

#complex y_receive for CVNN encoder
y_receive_cplx = y_receive.astype(np.complex64)

#targets: sin/cos encoding (real, for diffusion)
Y_sin = np.sin(target_azimuth).astype(np.float32)
Y_cos = np.cos(target_azimuth).astype(np.float32)
Y_all = np.hstack([Y_sin, Y_cos]).astype(np.float32)  # (N,8)

N = y_receive_cplx.shape[0]
idx = np.arange(N)
idx_tr, idx_te = train_test_split(idx, test_size=cfg.test_size, random_state=cfg.seed, shuffle=True)
Y_tr_cplx, Y_te_cplx = y_receive_cplx[idx_tr], y_receive_cplx[idx_te]
Y_tr, Y_te = Y_all[idx_tr], Y_all[idx_te]
TH_te = target_azimuth[idx_te].astype(np.float32)  # (N_te,4)

# Complex-valued dataset

class AngleDatasetCVNN(Dataset):
    def __init__(self, Y_cplx, Y_sincos):
        super().__init__()
        self.Y_cplx = torch.from_numpy(Y_cplx)      # (N,64) complex64
        self.Y_sincos = torch.from_numpy(Y_sincos)  # (N,8) float32
    def __len__(self):
        return self.Y_cplx.shape[0]
    def __getitem__(self, i):
        return self.Y_cplx[i], self.Y_sincos[i]

ds_tr = AngleDatasetCVNN(Y_tr_cplx, Y_tr)
ds_te = AngleDatasetCVNN(Y_te_cplx, Y_te)

dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
dl_te = DataLoader(ds_te, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

def get_beta_schedule(T, kind="cosine"):
    if kind == "linear":
        return torch.linspace(1e-4, 2e-2, T)
    elif kind == "cosine":
        steps = T + 1; s = 0.008
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-8, 0.999)
    else:
        raise ValueError("Unknown beta schedule")

betas = get_beta_schedule(cfg.betas_T, cfg.beta_schedule).to(cfg.device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=cfg.device), alphas_cumprod[:-1]], dim=0)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

class ComplexLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.Wr = nn.Linear(in_f, out_f)
        self.Wi = nn.Linear(in_f, out_f)
    def forward(self, xr, xi):
        yr = self.Wr(xr) - self.Wi(xi)
        yi = self.Wr(xi) + self.Wi(xr)
        return yr, yi

class ComplexBatchNorm1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.bn = nn.BatchNorm1d(ch * 2)
    def forward(self, xr, xi):
        x_cat = torch.cat([xr, xi], dim=1)  # (B, 2*ch)
        x_norm = self.bn(x_cat)             # normalize jointly
        xr_out = x_norm[:, :xr.size(1)]
        xi_out = x_norm[:, xr.size(1):]
        return xr_out, xi_out

def crelu(xr, xi):
    return F.relu(xr), F.relu(xi)

#complex encoder: y_receive (complex) to real conditioning vector
class CVNNEncoder(nn.Module):
    def __init__(self, M=16, cond_dim=256):
        super().__init__()
        # Input: (B, M) complex 
        self.fc1 = ComplexLinear(M, 128)
        self.bn1 = ComplexBatchNorm1d(128)
        self.fc2 = ComplexLinear(128, 256)
        self.bn2 = ComplexBatchNorm1d(256)
        self.fc3 = ComplexLinear(256, cond_dim)
        # Final real projection: concatenate real+imag 
        self.final = nn.Linear(cond_dim * 2, cond_dim)

    def forward(self, y_cplx):
        # y_cplx: (B,M) complex64
        xr = y_cplx.real  # (B,M)
        xi = y_cplx.imag  # (B,M)
        xr, xi = self.fc1(xr, xi); xr, xi = self.bn1(xr, xi); xr, xi = crelu(xr, xi)
        xr, xi = self.fc2(xr, xi); xr, xi = self.bn2(xr, xi); xr, xi = crelu(xr, xi)
        xr, xi = self.fc3(xr, xi); xr, xi = crelu(xr, xi)
        # Concatenate and project to real conditioning vector
        x_cat = torch.cat([xr, xi], dim=1) 
        cond = self.final(x_cat)            
        return cond
        
# Sinusoidal time embedding
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer("freqs", torch.exp(torch.linspace(math.log(1e-4), math.log(1.0), half)), persistent=False)
    def forward(self, t: torch.Tensor):
        x = (t.float() / max(1, cfg.betas_T - 1)).unsqueeze(-1)
        ang = 2.0 * math.pi * x * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb

class CondDenoiserHybrid(nn.Module):
    def __init__(self, y_dim=8, cond_dim=256, time_dim=128, hidden=512, layers=4):
        super().__init__()
        self.cvnn_encoder = CVNNEncoder(M=16, cond_dim=cond_dim)
        self.t_emb = SinusoidalTimeEmbedding(time_dim)
        self.t_proj = nn.Sequential(nn.Linear(time_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.c_proj = nn.Sequential(nn.Linear(cond_dim, hidden), nn.SiLU())
        self.y_in = nn.Sequential(nn.Linear(y_dim, hidden), nn.SiLU())
        blocks = []
        for _ in range(layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Linear(hidden, y_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y_t, y_cplx_cond, t):
        te = self.t_proj(self.t_emb(t))           
        ce = self.c_proj(self.cvnn_encoder(y_cplx_cond))  
        ye = self.y_in(y_t)  
        h = ye + te + ce
        h = self.blocks(h)
        eps = self.out(h)                         
        return eps

model = CondDenoiserHybrid(
    y_dim=8, cond_dim=cfg.cond_emb_dim, time_dim=cfg.time_emb_dim,
    hidden=cfg.hidden, layers=cfg.layers
).to(cfg.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

def safe_save(obj, path):
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_ckpt_", suffix=".pt")
    os.close(tmp_fd); torch.save(obj, tmp_path); os.replace(tmp_path, path)

last_checkpoint_state = {"epoch": 0}

def save_latest(model, epoch, hist_db_psavg):
    state = {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "nmse_test_hist_db_psavg": hist_db_psavg}
    safe_save(state, os.path.join(cfg.out_dir, "latest.pt"))
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump({"epoch": epoch, "nmse_test_hist_db_psavg": hist_db_psavg}, f)

def save_best(model, epoch, hist_db_psavg):
    state = {"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "nmse_test_hist_db_psavg": hist_db_psavg}
    safe_save(state, os.path.join(cfg.out_dir, "best.pt"))

def _graceful_exit_handler(signum, frame):
    save_latest(model, last_checkpoint_state["epoch"], last_checkpoint_state.get("hist_db_psavg", []))
signal.signal(signal.SIGINT, _graceful_exit_handler)
signal.signal(signal.SIGTERM, _graceful_exit_handler)
atexit.register(lambda: save_latest(model, last_checkpoint_state["epoch"], last_checkpoint_state.get("hist_db_psavg", [])))

def q_sample(y0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(y0)
    sqrt_ab = torch.sqrt(alphas_cumprod[t]).unsqueeze(-1)
    sqrt_omab = torch.sqrt(1.0 - alphas_cumprod[t]).unsqueeze(-1)
    return sqrt_ab * y0 + sqrt_omab * noise

@torch.no_grad()
def p_sample_step(model, y_cplx_cond, y_t, t, t_prev, ddim=True, eta=0.5):
    eps_pred = model(y_t, y_cplx_cond, t)
    a_t = alphas[t]; ab_t = alphas_cumprod[t]
    sqrt_one_minus_ab_t = torch.sqrt(1 - ab_t)
    y0_pred = (y_t - sqrt_one_minus_ab_t.unsqueeze(-1) * eps_pred) / torch.sqrt(ab_t).unsqueeze(-1)
    if ddim:
        ab_prev = alphas_cumprod[t_prev] if (t_prev >= 0).all() else torch.ones_like(ab_t)
        sigma_t = eta * torch.sqrt(((1.0 - ab_prev) / (1.0 - ab_t)) * (1.0 - (ab_t / ab_prev)))
        dir_term = torch.sqrt(ab_prev).unsqueeze(-1) * y0_pred
        noise_term = torch.sqrt(1.0 - ab_prev - sigma_t**2).unsqueeze(-1) * eps_pred
        z = torch.randn_like(y_t)
        return dir_term + noise_term + sigma_t.unsqueeze(-1) * z
    else:
        beta_t = betas[t]
        mean = torch.sqrt(1.0 / a_t).unsqueeze(-1) * (y_t - ((beta_t / torch.sqrt(1.0 - ab_t)).unsqueeze(-1) * eps_pred))
        var = posterior_variance[t].unsqueeze(-1)
        z = torch.randn_like(y_t)
        return mean + (t != 0).float().unsqueeze(-1) * torch.sqrt(var) * z

@torch.no_grad()
def sample_angles(model, y_cplx_cond, K=1, ddim=True, eta=0.5):
    model.eval()
    B = y_cplx_cond.shape[0]; T = cfg.betas_T; device = y_cplx_cond.device
    theta_out = []
    for _ in range(K):
        y_t = torch.randn(B, 8, device=device)
        for ti in reversed(range(T)):
            t = torch.full((B,), ti, device=device, dtype=torch.long)
            t_prev = torch.full((B,), max(ti - 1, 0), device=device, dtype=torch.long)
            y_t = p_sample_step(model, y_cplx_cond, y_t, t, t_prev, ddim=ddim, eta=eta)
        s, c = y_t[:, :4], y_t[:, 4:]
        norm = torch.sqrt(torch.clamp(s**2 + c**2, min=1e-12))
        theta = torch.atan2(s / norm, c / norm)
        theta_out.append(theta.unsqueeze(1))
    return torch.cat(theta_out, dim=1)

def train_one_epoch(model, dl, opt, scaler):
    model.train(); total = 0.0; count = 0
    for y_cplx, yb in dl:
        y_cplx = y_cplx.to(cfg.device)
        y0 = yb.to(cfg.device)
        B = y0.shape[0]
        t = torch.randint(0, cfg.betas_T, (B,), device=cfg.device, dtype=torch.long)
        noise = torch.randn_like(y0)
        y_t = q_sample(y0, t, noise)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=cfg.use_amp):
            eps_pred = model(y_t, y_cplx, t)
            loss = F.mse_loss(eps_pred, noise)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update()
        total += loss.item() * B; count += B
    return total / max(1, count)

@torch.no_grad()
@torch.no_grad()
def evaluate_rmse_db_psavg(model, dl, th_true_np, K=8, ddim=True, eta=0.5):
    Y_cplx_list = []
    for y_cplx, _ in dl:
        Y_cplx_list.append(y_cplx)
    Y_cplx = torch.cat(Y_cplx_list, dim=0).to(cfg.device)

    theta_samps = sample_angles(model, Y_cplx, K=K, ddim=ddim, eta=eta)
    s = torch.sin(theta_samps).mean(dim=1)
    c = torch.cos(theta_samps).mean(dim=1)
    norm = torch.sqrt(torch.clamp(s**2 + c**2, min=1e-12))
    theta_hat = torch.atan2(s / norm, c / norm).cpu().numpy()
    def circ_diff(a, b):
        d = a - b
        return (d + np.pi) % (2*np.pi) - np.pi

    diff_rad = circ_diff(theta_hat, th_true_np)
    mae_rad = float(np.mean(np.abs(diff_rad)))
    diff_deg = np.rad2deg(diff_rad)
    rmse_each_deg = np.sqrt(np.mean(diff_deg**2, axis=1))  
    rmse_db_each = 10.0 * np.log10(np.maximum(rmse_each_deg, cfg.db_eps))
    rmse_db_psavg = float(np.mean(rmse_db_each))
    rmse_deg_mean = float(np.mean(rmse_each_deg))
    return mae_rad, rmse_deg_mean, rmse_db_psavg, theta_hat


print(f"Device: {cfg.device}")
epoch_list = []; rmse_test_hist_db_psavg = []; best_mae = float("inf")

for epoch in range(1, cfg.epochs + 1):
    t0 = time.time()
    tr_loss = train_one_epoch(model, dl_tr, optimizer, scaler)
    mae_rad, rmse_deg_mean, rmse_db_psavg, _ = evaluate_rmse_db_psavg(model, dl_te, TH_te, K=cfg.num_samples_per_test, ddim=cfg.ddim, eta=cfg.ddim_eta)
    dt = time.time() - t0
    epoch_list.append(epoch); rmse_test_hist_db_psavg.append(rmse_db_psavg)
    last_checkpoint_state["epoch"] = epoch; last_checkpoint_state["hist_db_psavg"] = rmse_test_hist_db_psavg
    print(f"[{epoch:03d}] loss={tr_loss:.6f} | RMSE(deg)={rmse_deg_mean:.3f} RMSE_dB(ps-avg)={rmse_db_psavg:.2f} | {dt:.1f}s")
    save_latest(model, epoch, rmse_test_hist_db_psavg)
    if mae_rad < best_mae:
        best_mae = mae_rad; save_best(model, epoch, rmse_test_hist_db_psavg)
        print(f"Saved BEST checkpoint at epoch {epoch} with MAE(rad)={best_mae:.5f}")
    if epoch % cfg.save_every == 0:
        safe_save({"model": model.state_dict(), "cfg": cfg.__dict__, "epoch": epoch, "rmse_test_hist_db_psavg": rmse_test_hist_db_psavg},
                  os.path.join(cfg.out_dir, f"epoch_{epoch:04d}.pt"))

# Plot
epochs_arr = np.array(epoch_list); rmse_db_psavg_arr = np.array(rmse_test_hist_db_psavg)
plt.figure(figsize=(7.5, 4.5))
plt.plot(epochs_arr, rmse_db_psavg_arr, '-o', lw=1.8, ms=4)
plt.xlabel('Epoch'); plt.ylabel('Angle RMSE (dB) [per-sample→dB→mean]')
plt.title('CVNN-Diffusion: Test Angle RMSE (dB) vs Epochs'); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(cfg.out_dir, 'cvnn_diffusion_rmse_db_vs_epochs.png'), dpi=180, bbox_inches='tight')
plt.show()