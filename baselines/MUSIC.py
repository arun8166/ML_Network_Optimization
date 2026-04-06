#MUSIC implementation

import numpy as np
import mat73
import matplotlib.pyplot as plt
import scipy.io

file_path = ''
d = scipy.io.loadmat(file_path)
y_receive = d["y_receive"] #(64,100,6,5000)
target_azimuth = d["target_azimuth"]
target_azimuth = np.squeeze(target_azimuth).T #(5000,4)

from scipy.signal import find_peaks

def music_1d_peaks(y_receive, n_targets=4, n_peaks=4, n_grid=128, A=None, grid=None): #main function
    y = np.asarray(y_receive)
    M, L = y.shape
    R = (y @ y.conj().T) / L
    w, V = np.linalg.eigh(R)
    idx = np.argsort(w)[::-1]
    V = V[:, idx]
    En = V[:, n_targets:]
    X = En.conj().T @ A
    denom = np.sum(np.abs(X) ** 2, axis=0)
    denom = np.maximum(denom, 1e-12)
    P = 1.0 / denom
    P_db = 10 * np.log10(P / np.max(P))
    min_dist = max(1, n_grid // (12 * n_peaks))
    peaks, _ = find_peaks(P_db, distance=min_dist)
    if peaks.size == 0:
        sel = np.argsort(P_db)[-n_peaks:]
    else:
        sel = peaks[np.argsort(P_db[peaks])[-min(n_peaks, peaks.size):]]
        if sel.size < n_peaks:
            extra = np.setdiff1d(np.argsort(P_db)[::-1], sel, assume_unique=False)[:(n_peaks - sel.size)]
            sel = np.concatenate([sel, extra])
    sel = np.sort(sel)
    return grid[sel]

def greedy_minabs_match(est_angles, target_azimuth): #match angle predictions with the ground-truth values
    est = np.asarray(est_angles).reshape(-1)
    tgt = np.asarray(target_azimuth).reshape(-1)
    C = np.abs(est[:, None] - tgt[None, :])
    rows = list(range(C.shape[0]))
    cols = list(range(C.shape[1]))
    pairs = []
    while len(rows) > 0:
        sub = C[np.ix_(rows, cols)]
        k = np.argmin(sub)
        r_sub, c_sub = np.unravel_index(k, sub.shape)
        r = rows.pop(r_sub)
        c = cols.pop(c_sub)
        pairs.append((r, c))
    est_reordered = np.empty_like(tgt)
    for r, c in pairs:
        est_reordered[c] = est[r]
    return est_reordered

def nmse_db(pred, tgt, eps=1e-12):
    pred = np.asarray(pred).reshape(-1)
    tgt = np.asarray(tgt).reshape(-1)
    num = np.sum((pred - tgt) ** 2)
    den = np.sum(tgt ** 2)
    nmse = num / max(den, eps)
    return 10.0 * np.log10(max(nmse, eps))

N = y_receive.shape[-1]
n_grid = 3600
M = y_receive.shape[0]

grid = np.linspace(-np.pi / 2, np.pi / 2, n_grid)
m = np.arange(M)
k = 2 * np.pi * 0.5 / 1.0
A = np.exp(1j * k * np.outer(m, np.sin(grid)))

snr_levels_db = np.array([-20, -10, 0, 10, 20, 30], dtype=int)
avg_nmse_db = np.empty(snr_levels_db.size, dtype=float)

plt.figure()
for j, snr in enumerate(snr_levels_db):
    nmse_vals_db = np.empty(N, dtype=float)
    for i in range(N):
        est_angles = music_1d_peaks(y_receive[:, :, j, i], n_targets=4, n_peaks=4, n_grid=n_grid, A=A, grid=grid)
        est_ord = greedy_minabs_match(est_angles, target_azimuth[i])
        nmse_vals_db[i] = nmse_db(est_ord, target_azimuth[i])
    avg_nmse_db[j] = float(np.mean(nmse_vals_db))

    x = np.sort(nmse_vals_db)
    y = (np.arange(1, x.size + 1) / x.size)
    plt.step(x, y, where="post", label=f"{snr} dB")

plt.xlabel("NMSE (dB)")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.show()

print("SNR(dB):", snr_levels_db)
print("Avg NMSE(dB):", avg_nmse_db)