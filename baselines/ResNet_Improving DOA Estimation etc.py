#ResNet model introduced in the paper: Improving DOA Estimation via an Optimal Deep Residual Neural Network Classifier on Uniform Linear Arrays
import os
import random
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

mat_path = "" #enter path
d = scipy.io.loadmat(mat_path)

y_receive = d["y_receive"][:, :, 5, :]          
target_azimuth = np.squeeze(d["target_azimuth"]).T  
N = y_receive.shape[-1]  
M = 16
T = y_receive.shape[1]   

cov_c = np.zeros((M, M, N), dtype=np.complex64)
for i in range(N):
    y = np.asarray(y_receive[:, :, i])  
    R = (y @ y.conj().T) / T
    tr = np.trace(R).real
    R = R / (tr + 1e-8)
    cov_c[:, :, i] = R


indices = np.round(target_azimuth * 180.0 / np.pi + 90.0).astype(np.int64)  
indices = np.clip(indices, 0, 180)
cov = np.stack((cov_c.real, cov_c.imag), axis=0).astype(np.float32) 
X = np.transpose(cov, (3, 0, 1, 2))                                  
y_cls = indices                                                     

perm = np.random.permutation(N)
n_train = int(0.8 * N)
tr_idx, va_idx = perm[:n_train], perm[n_train:]
X_tr, y_tr = X[tr_idx], y_cls[tr_idx]
X_va, y_va = X[va_idx], y_cls[va_idx]

class CovDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)             
        self.y = torch.from_numpy(y)            
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dl_tr = DataLoader(CovDataset(X_tr, y_tr), batch_size=64, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
dl_va = DataLoader(CovDataset(X_va, y_va), batch_size=256, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        p = k // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.skip = None
        if in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            identity = self.skip(identity)
        out = F.relu(out + identity)
        return out

class DOAResNet(nn.Module):
    def __init__(self, num_targets=4, num_classes=181):
        super().__init__()
        self.num_targets = num_targets
        self.num_classes = num_classes

        self.stem = nn.Sequential(
            nn.Conv2d(2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(512, 256, k=3)
        self.res2 = ResidualBlock(256, 128, k=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_targets * num_classes)

    def forward(self, x):
        x = self.stem(x)          
        x = self.res1(x)          
        x = self.res2(x)          
        x = self.pool(x).flatten(1) 
        x = self.fc(x)            
        x = x.view(-1, self.num_targets, self.num_classes) 
        return x

model = DOAResNet(num_targets=4, num_classes=181).to(device)
print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")

def multihead_ce_loss(logits, y):
    loss = 0.0
    for k in range(logits.shape[1]):
        loss = loss + F.cross_entropy(logits[:, k, :], y[:, k])
    return loss / logits.shape[1]

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

use_amp = (device == "cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    correct_per_head = np.zeros(4, dtype=np.int64)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = multihead_ce_loss(logits, yb)
        pred = logits.argmax(dim=-1)  # (B,4)
        for k in range(4):
            correct_per_head[k] += (pred[:, k] == yb[:, k]).sum().item()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total += bs

    avg_loss = total_loss / max(total, 1)
    acc_heads = correct_per_head / max(total, 1)
    acc_mean = acc_heads.mean()
    return avg_loss, acc_heads, acc_mean

best_va = 1e9
save_path = "doa_resnet.pt"

epochs = 100
for epoch in range(1, epochs + 1):
    model.train()
    running = 0.0
    seen = 0

    for xb, yb in dl_tr:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = multihead_ce_loss(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        bs = xb.size(0)
        running += loss.item() * bs
        seen += bs

    tr_loss = running / max(seen, 1)
    va_loss, va_acc_heads, va_acc_mean = evaluate(model, dl_va)
    print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} | val loss {va_loss:.4f} | val acc mean {va_acc_mean:.4f}")

    if va_loss < best_va:
        best_va = va_loss
        torch.save(
            {"model": model.state_dict(),
             "opt": opt.state_dict(),
             "epoch": epoch,
             "best_va": best_va},
            save_path
        )

print("Saved best model to:", save_path)

@torch.no_grad()
def predict_angles_rad(model, X_np):
    model.eval()
    X_t = torch.from_numpy(X_np).to(device)
    logits = model(X_t)                 
    idx = logits.argmax(dim=-1).cpu().numpy()  
    deg = idx.astype(np.float32) - 90.0
    rad = deg * np.pi / 180.0
    return rad

def batch_rmse_db_from_class_indices(pred_idx, true_idx, eps=1e-12):
    pred_deg = pred_idx.float() - 90.0
    true_deg = true_idx.float() - 90.0
    err = pred_deg - true_deg                     
    rmse_deg = torch.sqrt(torch.mean(err**2, dim=1)) 
    rmse_db = 10.0 * torch.log10(rmse_deg + eps)      
    return rmse_deg, rmse_db

@torch.no_grad()
def evaluate_with_rmse_db(model, loader):
    model.eval()
    total_loss = 0.0
    total = 0
    sum_rmse_db = 0.0
    sum_rmse_deg = 0.0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)   

        logits = model(xb)                      
        loss = multihead_ce_loss(logits, yb)
        pred_idx = logits.argmax(dim=-1)        
        rmse_deg, rmse_db = batch_rmse_db_from_class_indices(pred_idx, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        sum_rmse_deg += rmse_deg.sum().item()
        sum_rmse_db  += rmse_db.sum().item()
        total += bs

    avg_loss = total_loss / max(total, 1)
    avg_rmse_deg = sum_rmse_deg / max(total, 1)
    avg_rmse_db  = sum_rmse_db  / max(total, 1)  
    return avg_loss, avg_rmse_deg, avg_rmse_db

ckpt = torch.load("doa_resnet.pt", map_location=device)
model.load_state_dict(ckpt["model"])

test_loss, test_rmse_deg, test_rmse_db = evaluate_with_rmse_db(model, dl_va)
print(f"Test loss: {test_loss:.4f}")   
print(f"RMSE (dB): {test_rmse_db:.4f}")
