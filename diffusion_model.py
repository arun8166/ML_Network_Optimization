import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy.io
import math

def complex_to_real(complex_array):
    """Convert complex array to [real, imag] representation"""
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)
    return np.concatenate([real_part, imag_part], axis=1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AngleDiffusionModel(nn.Module):
    def __init__(self, input_dim=128, output_dim=2, hidden_dim=256, time_dim=64):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input condition encoder
        self.input_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Main denoising network
        self.main_net = nn.Sequential(
            nn.Linear(hidden_dim + output_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x_cond, y_noisy, timestep):
        # Get embeddings
        t_emb = self.time_mlp(timestep)
        x_emb = self.input_encoder(x_cond)
        
        # Combine condition and time embeddings
        combined_emb = x_emb + t_emb
        
        # Concatenate with noisy target
        net_input = torch.cat([combined_emb, y_noisy], dim=1)
        
        # Predict the noise
        predicted_noise = self.main_net(net_input)
        
        return predicted_noise

class DiffusionTrainer:
    def __init__(self, model, device, num_timesteps=50):
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        
        # Create noise schedule (cosine schedule works better)
        self.betas = self.cosine_beta_schedule(num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=20, factor=0.8, verbose=True
        )
        
    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine noise schedule"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        """Forward process: add noise to x_start at timestep t"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        return (
            sqrt_alphas_cumprod_t.reshape(-1, 1) * x_start +
            sqrt_one_minus_alphas_cumprod_t.reshape(-1, 1) * noise
        )
    
    def p_losses(self, x_cond, x_start, t, noise=None):
        """Calculate training loss"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        predicted_noise = self.model(x_cond, x_noisy, t)
        
        # Calculate loss
        loss = F.mse_loss(noise, predicted_noise)
        return loss
    
    def train_step(self, x_batch, y_batch):
        """Single training step"""
        batch_size = x_batch.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
        
        # Calculate loss
        loss = self.p_losses(x_batch, y_batch, t)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

def p_sample(model, x_cond, x, t, betas, alphas, alphas_cumprod):
    """Single denoising step"""
    betas_t = betas[t].reshape(-1, 1)
    sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1. - alphas_cumprod[t]).reshape(-1, 1)
    sqrt_recip_alphas_t = torch.sqrt(1.0 / alphas[t]).reshape(-1, 1)
    
    # Predict noise
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x_cond, x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t[0] == 0:
        return model_mean
    else:
        posterior_variance_t = betas[t].reshape(-1, 1)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def sample_angles(model, x_cond, trainer, num_samples=1):
    """Generate angle predictions using reverse diffusion"""
    model.eval()
    device = next(model.parameters()).device
    
    with torch.no_grad():
        # Start from pure noise
        batch_size = x_cond.shape[0]
        x = torch.randn(batch_size, 2, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(trainer.num_timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = p_sample(
                model, x_cond, x, t,
                trainer.betas, trainer.alphas, trainer.alphas_cumprod
            )
    
    model.train()
    return x

def train_diffusion_model(X_complex, y_angles, epochs=2000, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # FIXED: Convert complex data to real BEFORE any scaling
    X = complex_to_real(X_complex)
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y_angles)
    
    # Train/validation split
    train_size = int(0.9 * len(X))
    indices = np.random.permutation(len(X))
    
    X_train = torch.FloatTensor(X_norm[indices[:train_size]]).to(device)
    y_train = torch.FloatTensor(y_norm[indices[:train_size]]).to(device)
    X_val = torch.FloatTensor(X_norm[indices[train_size:]]).to(device)
    y_val = torch.FloatTensor(y_norm[indices[train_size:]]).to(device)
    
    # Initialize model and trainer
    model = AngleDiffusionModel(input_dim=X.shape[1])
    trainer = DiffusionTrainer(model, device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Data shape: X={X.shape}, y={y_angles.shape}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        # Training loop
        indices = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            loss = trainer.train_step(x_batch, y_batch)
            train_losses.append(loss)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                x_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]
                
                batch_size_val = x_batch.shape[0]
                t = torch.randint(0, trainer.num_timesteps, (batch_size_val,), device=device).long()
                val_loss = trainer.p_losses(x_batch, y_batch, t).item()
                val_losses.append(val_loss)
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        trainer.scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_diffusion_model.pt')
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_diffusion_model.pt', weights_only=False))
    
    return model, trainer, scaler_X, scaler_y

def evaluate_model(model, trainer, X_test_complex, y_test, scaler_X, scaler_y):
    """FIXED: Evaluate the diffusion model with proper complex data handling"""
    device = next(model.parameters()).device
    
    # FIXED: Convert complex test data to real format first
    X_test_real = complex_to_real(X_test_complex)
    
    # Then apply scaling
    X_test_norm = scaler_X.transform(X_test_real)
    X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
    
    # Generate predictions
    predictions = sample_angles(model, X_test_tensor, trainer)
    predictions_denorm = scaler_y.inverse_transform(predictions.cpu().numpy())
    
    return predictions_denorm

# Main execution
if __name__ == "__main__":
    # Load data
    mat1 = scipy.io.loadmat('/kaggle/input/angle-estimate-azimuth-and-elevation-from-y-rec/1_BS_1_target_1000_data.mat')
    target_azimuth1 = mat1['target_azimuth'].reshape(1000, 1)
    target_elevation1 = mat1['target_elevation'].reshape(1000, 1)
    y_receive1 = mat1['y_receive'].T  # Complex antenna data
    
    y_angles = np.concatenate([target_azimuth1, target_elevation1], axis=1)
    
    print(f"Loaded data: X={y_receive1.shape} (complex), y={y_angles.shape}")
    
    # Train the diffusion model
    model, trainer, scaler_X, scaler_y = train_diffusion_model(
        y_receive1, y_angles, epochs=2000, batch_size=32
    )
    
    # FIXED: Test predictions on first 10 samples with proper complex handling
    test_X_complex = y_receive1[:10]  # Keep as complex for proper conversion
    test_predictions = evaluate_model(model, trainer, test_X_complex, y_angles[:10], scaler_X, scaler_y)
    
    print("\n DIFFUSION MODEL RESULTS:")
    within_5pct = 0
    
    for i in range(10):
        pred_az, pred_el = test_predictions[i]
        true_az, true_el = y_angles[i]
        
        error = np.sqrt((pred_az - true_az)**2 + (pred_el - true_el)**2)
        true_mag = max(np.sqrt(true_az**2 + true_el**2), 0.1)
        rel_error = (error / true_mag) * 100
        
        if rel_error < 5.0:
            within_5pct += 1
            status = "great"
        elif rel_error < 10.0:
            status = "good"
        else:
            status = "bad"
            
        print(f"Sample {i+1}: Pred=({pred_az:.3f}°, {pred_el:.3f}°), True=({true_az:.3f}°, {true_el:.3f}°) {status}")
        print(f"  Error={error:.3f}°, Rel Error={rel_error:.1f}%")
    
    # Full dataset evaluation
    all_predictions = evaluate_model(model, trainer, y_receive1, y_angles, scaler_X, scaler_y)
    
    # Calculate comprehensive metrics
    overall_rmse = np.sqrt(np.mean((all_predictions - y_angles)**2))
    
    relative_errors = []
    for i in range(len(y_angles)):
        true_mag = max(np.sqrt(np.sum(y_angles[i]**2)), 0.1)
        pred_error = np.sqrt(np.sum((all_predictions[i] - y_angles[i])**2))
        rel_error = pred_error / true_mag
        relative_errors.append(rel_error)
    
    within_1_percent = np.mean(np.array(relative_errors) < 0.01) * 100
    within_3_percent = np.mean(np.array(relative_errors) < 0.03) * 100  
    within_5_percent = np.mean(np.array(relative_errors) < 0.05) * 100
    within_10_percent = np.mean(np.array(relative_errors) < 0.10) * 100
    
    print(f"\n COMPREHENSIVE EVALUATION:")
    print(f"Overall RMSE: {overall_rmse:.4f}°")
    print(f"Predictions within 1% accuracy: {within_1_percent:.1f}%")
    print(f"Predictions within 3% accuracy: {within_3_percent:.1f}%") 
    print(f"Predictions within 5% accuracy: {within_5_percent:.1f}%")
    print(f"Predictions within 10% accuracy: {within_10_percent:.1f}%")
    
    if within_5_percent >= 50:
        print(" TARGET ACHIEVED: >50% within 5% accuracy!")
    else:
        print(f" Progress: {within_5_percent:.1f}% / 50% target")
        
    print(" Diffusion model training complete!")
