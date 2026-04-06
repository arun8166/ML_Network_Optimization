#ESPRIT implementation

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from numpy.linalg import eigh, pinv

file_path = ''
d = scipy.io.loadmat(file_path)

y_receive = d["y_receive"]            # (64, 100, 6, 5000)
target_azimuth = d["target_azimuth"]  
target_azimuth = np.squeeze(target_azimuth).T  


def esprit_1d(y_snapshots, n_targets=4, k=None, eps=1e-12): #main function
    Y = np.asarray(y_snapshots)
    M, L = Y.shape
    R = (Y @ Y.conj().T) / max(L, 1)

    w, V = eigh(R)
    idx = np.argsort(w)[::-1]  
    V = V[:, idx]
    Es = V[:, :n_targets]     
    Es1 = Es[:-1, :]          
    Es2 = Es[1:, :]            
    Psi = pinv(Es1) @ Es2     

    evals, _ = np.linalg.eig(Psi)
    phases = np.angle(evals)   
    sin_theta = phases / max(k, eps)
    sin_theta = np.clip(sin_theta, -1.0, 1.0)
    angles = np.arcsin(sin_theta)
    angles = np.sort(np.real(angles))
    return angles


def greedy_minabs_match(est_angles, target_azimuth): # match predictions with ground-truth values
    est = np.asarray(est_angles).reshape(-1)
    tgt = np.asarray(target_azimuth).reshape(-1)

    C = np.abs(est[:, None] - tgt[None, :])
    rows = list(range(C.shape[0]))
    cols = list(range(C.shape[1]))
    pairs = []
    while len(rows) > 0 and len(cols) > 0:
        sub = C[np.ix_(rows, cols)]
        kmin = np.argmin(sub)
        r_sub, c_sub = np.unravel_index(kmin, sub.shape)
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
snr_levels_db = np.array([-20, -10, 0, 10, 20, 30], dtype=int)
avg_nmse_db = np.empty(snr_levels_db.size, dtype=float)
M = y_receive.shape[0]
k = 2 * np.pi * 0.5 / 1.0  

#plot the CDF's at different SNR's
plt.figure()
for j, snr in enumerate(snr_levels_db):
    nmse_vals_db = np.empty(N, dtype=float)
    for i in range(N):
        est_angles = esprit_1d(y_receive[:, :, j, i], n_targets=4, k=k)
        est_ord = greedy_minabs_match(est_angles, target_azimuth[i])
        nmse_vals_db[i] = nmse_db(est_ord, target_azimuth[i])

    avg_nmse_db[j] = float(np.mean(nmse_vals_db))

    x = np.sort(nmse_vals_db)
    y_cdf = (np.arange(1, x.size + 1) / x.size)
    plt.step(x, y_cdf, where="post", label=f"{snr} dB")

plt.xlabel("NMSE (dB)")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.show()

print("SNR(dB):", snr_levels_db)
print("Avg NMSE(dB):", avg_nmse_db)
