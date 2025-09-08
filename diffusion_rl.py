import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from sklearn.preprocessing import StandardScaler
from torch.distributions import Normal
import scipy.io
import time
from torch.cuda.amp import autocast, GradScaler

def complex_to_real(complex_array):
    """Convert complex array to [real, imag] representation"""
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)
    return np.concatenate([real_part, imag_part], axis=1)

# Diffusion Model Architecture
class AngleRegressionDiffusion(nn.Module):
    def __init__(self, input_dim=128, output_dim=2, hidden_dim=512, num_layers=6):
        super().__init__()
        
        # Time embedding with sinusoidal encoding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input feature encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )
        
        # Residual blocks for better gradient flow
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output layers for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.log_std_head = nn.Linear(hidden_dim, output_dim)
        
    def get_time_embedding(self, timesteps, dim):
        """Sinusoidal time embeddings"""
        half = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb).to(timesteps.device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb
        
    def forward(self, x_input, y_noisy, timesteps):
        # Time embedding
        t_emb = self.get_time_embedding(timesteps, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        
        # Input encoding
        x_feat = self.input_encoder(x_input)
        
        # Combine input features and time
        h = x_feat + t_emb
        
        # Residual processing
        for block in self.residual_blocks:
            h = block(h)
        
        # Output mean and log_std for action distribution
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, -5, 2)  # Prevent extreme values
        
        return mean, log_std

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.block(x)

# DDPO Trainer - Optimized for GPU
class DDPOTrainer:
    def __init__(self, model, device, num_timesteps=50, beta_start=0.0001, beta_end=0.02, 
                 gamma=0.99, clip_epsilon=0.2, kl_penalty=0.1, lr=1e-4):
        self.device = device
        self.model = model.to(device)
        self.reference_model = copy.deepcopy(model).to(device)
        self.num_timesteps = num_timesteps
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.kl_penalty = kl_penalty
        
        # Freeze reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # Cosine noise schedule - move to device
        self.betas = self.cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = (1 - self.betas).to(device)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        # Mixed precision training
        self.scaler = GradScaler()
        
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule - often better than linear"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def sample_trajectory(self, x_batch, y_batch):
        """Generate complete denoising trajectory"""
        batch_size = x_batch.shape[0]
        
        # Start from noise
        y_t = torch.randn_like(y_batch)
        
        # Store trajectory information
        trajectory = {
            'log_probs': [],
            'ref_log_probs': []
        }
        
        # Reverse diffusion process
        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((batch_size,), float(t), dtype=torch.float32, device=self.device)
            
            # Use mixed precision for forward pass
            with autocast():
                # Current policy
                mean, log_std = self.model(x_batch, y_t, timesteps)
                std = torch.exp(log_std)
                policy_dist = Normal(mean, std)
                
                # Reference policy
                with torch.no_grad():
                    ref_mean, ref_log_std = self.reference_model(x_batch, y_t, timesteps)
                    ref_std = torch.exp(ref_log_std)
                    ref_dist = Normal(ref_mean, ref_std)
            
            # Sample action
            if t > 0:
                y_t_next = policy_dist.rsample()
            else:
                y_t_next = mean
            
            # Store log probabilities
            trajectory['log_probs'].append(policy_dist.log_prob(y_t_next).sum(dim=1))
            trajectory['ref_log_probs'].append(ref_dist.log_prob(y_t_next).sum(dim=1))
            
            y_t = y_t_next
            
        return trajectory, y_t
    
    def train_step(self, x_batch, y_batch):
        """Single DDPO training step with mixed precision"""
        # Generate trajectory
        trajectory, final_predictions = self.sample_trajectory(x_batch, y_batch)
        
        # Terminal reward based on RMSE
        terminal_reward = -torch.sqrt(torch.mean((final_predictions - y_batch) ** 2, dim=1))
        
        # PPO loss computation
        total_loss = 0
        total_kl = 0
        
        for i in range(len(trajectory['log_probs'])):
            log_probs = trajectory['log_probs'][i]
            old_log_probs = log_probs.detach()
            ref_log_probs = trajectory['ref_log_probs'][i]
            
            # PPO ratio and clipped objective
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * terminal_reward
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * terminal_reward
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # KL penalty
            kl_div = (log_probs - ref_log_probs).mean()
            
            total_loss += policy_loss + self.kl_penalty * kl_div
            total_kl += kl_div.item()
        
        # Mixed precision optimization
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        # Compute RMSE
        rmse = torch.sqrt(torch.mean((final_predictions - y_batch) ** 2))
        
        return {
            'loss': total_loss.item(),
            'rmse': rmse.item(),
            'reward': terminal_reward.mean().item(),
            'kl': total_kl / len(trajectory['log_probs'])
        }

# Training pipeline - GPU optimized
def train_ddpo_angle_prediction(X_complex, y_angles, epochs=200, batch_size=64):
    """Main training function optimized for GPU"""
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data preprocessing
    X = complex_to_real(X_complex)
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y_angles)
    
    # Convert to tensors and move to GPU
    X_tensor = torch.FloatTensor(X_norm).to(device)
    y_tensor = torch.FloatTensor(y_norm).to(device)
    
    # Initialize model and trainer
    model = AngleRegressionDiffusion(input_dim=X.shape[1], output_dim=2)
    trainer = DDPOTrainer(model, device, num_timesteps=50)
    
    # Training history
    history = {'loss': [], 'rmse': [], 'reward': [], 'kl': []}
    
    print("Starting DDPO Training...")
    print(f"Data shape: X={X.shape}, y={y_angles.shape}")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_metrics = {'loss': [], 'rmse': [], 'reward': [], 'kl': []}
        
        # Create random batches
        indices = torch.randperm(len(X_tensor), device=device)
        
        for i in range(0, len(X_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]
            
            # DDPO training step
            metrics = trainer.train_step(x_batch, y_batch)
            
            # Collect metrics
            for key in epoch_metrics:
                epoch_metrics[key].append(metrics[key])
        
        # Average metrics for epoch
        for key in history:
            history[key].append(np.mean(epoch_metrics[key]))
        
        # Print progress (less frequent for performance)
        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}: Loss={history['loss'][-1]:.4f}, "
                  f"RMSE={history['rmse'][-1]:.4f}, "
                  f"Reward={history['reward'][-1]:.4f}, "
                  f"KL={history['kl'][-1]:.4f}, "
                  f"Time: {elapsed:.1f}s")
            
            # Memory usage
            if device.type == 'cuda':
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB / {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    return model, trainer, scaler_X, scaler_y, history

# Inference function - GPU optimized
def predict_angles(model, x_input, device, num_inference_steps=50):
    """Generate predictions using trained DDPO model"""
    model.eval()
    x_input = x_input.to(device)
    
    with torch.no_grad():
        batch_size = x_input.shape[0]
        y_t = torch.randn(batch_size, 2, device=device)
        
        # Reverse diffusion process with mixed precision
        for t in reversed(range(num_inference_steps)):
            timesteps = torch.full((batch_size,), float(t), dtype=torch.float32, device=device)
            
            with autocast():
                mean, log_std = model(x_input, y_t, timesteps)
            
            if t > 0:
                std = torch.exp(log_std)
                noise = torch.randn_like(y_t)
                y_t = mean + std * noise
            else:
                y_t = mean
    
    model.train()
    return y_t

# Main execution
if __name__ == "__main__":
    # Load data
    mat1 = scipy.io.loadmat('/kaggle/input/angle-estimate-azimuth-and-elevation-from-y-rec/1_BS_1_target_1000_data.mat')
    target_azimuth1 = mat1['target_azimuth'].reshape(1000, 1)
    target_elevation1 = mat1['target_elevation'].reshape(1000, 1)
    y_receive1 = mat1['y_receive'].T
    
    # Combine targets
    y_angles = np.concatenate([target_azimuth1, target_elevation1], axis=1)
    
    # Train the model with GPU optimization
    model, trainer, scaler_X, scaler_y, history = train_ddpo_angle_prediction(
        y_receive1, y_angles, epochs=200, batch_size=64  # Increased batch size for P100
    )
    
    # Save the trained model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'history': history
    }, 'ddpo_angle_model.pt')
    
    print("Model saved as 'ddpo_angle_model.pt'")
    
    # Make predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_X = complex_to_real(y_receive1[:10])
    test_X_norm = scaler_X.transform(test_X)
    test_predictions = predict_angles(model, torch.FloatTensor(test_X_norm), device)
    test_predictions_denorm = scaler_y.inverse_transform(test_predictions.cpu().numpy())
    
    print("\nSample Predictions vs Ground Truth:")
    for i in range(5):
        pred_az, pred_el = test_predictions_denorm[i]
        true_az, true_el = y_angles[i]
        print(f"Sample {i+1}: Pred=({pred_az:.2f}°, {pred_el:.2f}°), "
              f"True=({true_az:.2f}°, {true_el:.2f}°)")
    
    # Calculate overall performance
    all_test_X = complex_to_real(y_receive1)
    all_test_X_norm = scaler_X.transform(all_test_X)
    all_predictions = predict_angles(model, torch.FloatTensor(all_test_X_norm), device)
    all_predictions_denorm = scaler_y.inverse_transform(all_predictions.cpu().numpy())
    
    # Compute RMSE
    rmse_az = np.sqrt(np.mean((all_predictions_denorm[:, 0] - y_angles[:, 0])**2))
    rmse_el = np.sqrt(np.mean((all_predictions_denorm[:, 1] - y_angles[:, 1])**2))
    overall_rmse = np.sqrt(np.mean((all_predictions_denorm - y_angles)**2))
    
    print(f"\nFinal Performance:")
    print(f"Azimuth RMSE: {rmse_az:.4f}°")
    print(f"Elevation RMSE: {rmse_el:.4f}°")
    print(f"Overall RMSE: {overall_rmse:.4f}°")
