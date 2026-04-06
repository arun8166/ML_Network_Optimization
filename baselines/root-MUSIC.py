#root-MUSIC implementation
import numpy as np
import scipy.io

d = scipy.io.loadmat('') #dataset path
y_receive = d['y_receive'][:,:,5,:] #(16,100,5000)
target_azimuth = np.squeeze(d["target_azimuth"]).T  #(5000,4)

def root_music_ula_rad(y_receive, K, d_over_lambda=0.5, forward_backward=True): #main function
    M, L, N = y_receive.shape
    est = np.full((N, K), np.nan, float)
    J = np.fliplr(np.eye(M))  
    for n in range(N):
        Y = y_receive[:, :, n]
        R = (Y @ Y.conj().T) / L  
        if forward_backward:
            R = 0.5 * (R + J @ R.conj() @ J)  

        evals, evecs = np.linalg.eigh(R)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        Un = evecs[:, K:]
        Gn = Un @ Un.conj().T
        c = np.array([np.trace(Gn, offset=lag) for lag in range(-(M-1), M)], dtype=complex)
        a = c[::-1] 
        roots = np.roots(a)

        inside = roots[np.abs(roots) < 1.0]
        cand = inside if inside.size >= K else roots
        chosen = cand[np.argsort(np.abs(np.abs(cand) - 1.0))[:K]]
        phi = np.angle(chosen)  
        mu = phi / (2*np.pi*d_over_lambda)  
        mu = np.clip(mu.real, -1.0, 1.0)
        theta = np.arcsin(mu) 
        est[n] = np.sort(theta)
    return est

def avg_10log10_rmse_deg(est_rad, gt_rad, eps=1e-12):
    est = np.sort(est_rad, axis=1)
    gt  = np.sort(gt_rad,  axis=1)
    rmse_rad = np.sqrt(np.mean((est - gt)**2, axis=1))
    rmse_deg = np.degrees(rmse_rad)
    score_per_sample = 10.0 * np.log10(np.maximum(rmse_deg, eps))
    return score_per_sample.mean(), score_per_sample, rmse_deg

y_receive = d['y_receive'][:,:,2,:]                 # (16,100,5000)
gt = np.squeeze(d["target_azimuth"]).T              # (5000,4) 
K = gt.shape[1]
est = root_music_ula_rad(y_receive, K, d_over_lambda=0.5, forward_backward=True)

avg_score, per_sample_score, rmse_deg = avg_10log10_rmse_deg(est, target_azimuth)
print("Avg metric:", avg_score)

