"""
Complete example showing how to generate CSI data and train ST-ResNet for prediction
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from generate_csi import generate_spatial_temporal_correlated_csi, csi_to_tensor_format
from st_resnet import STResNet, CSIDataset, train_model, evaluate_model

def main():
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Parameters
    num_symbols = 1000      # Total time samples
    num_users = 4          # Number of users/antennas
    k_factor_dB = 6.0      # Rician K-factor
    rho = 0.95             # Temporal correlation
    history_length = 5     # Number of past samples for prediction
    
    print("=== CSI Data Generation ===")
    
    # Generate CSI data
    H = generate_spatial_temporal_correlated_csi(
        num_symbols=num_symbols,
        num_users=num_users,
        k_factor_dB=k_factor_dB,
        rho=rho
    )
    
    print(f"Generated CSI shape: {H.shape}")
    print(f"Mean CSI magnitude: {np.mean(np.abs(H)):.4f}")
    
    # Convert to tensor format (real/imaginary separation)
    tensor_data = csi_to_tensor_format(H)
    print(f"Tensor data shape: {tensor_data.shape}")
    
    # Normalize data (important for neural network training)
    mean_val = np.mean(tensor_data)
    std_val = np.std(tensor_data)
    tensor_data = (tensor_data - mean_val) / std_val
    
    print(f"Normalized data - Mean: {np.mean(tensor_data):.6f}, Std: {np.std(tensor_data):.6f}")
    
    print("\n=== Dataset Preparation ===")
    
    # Create dataset
    dataset = CSIDataset(tensor_data, history_length=history_length)
    print(f"Dataset size: {len(dataset)} samples")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    print("\n=== Model Training ===")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = STResNet(
        in_channels=2,
        history_length=history_length,
        num_blocks=3,
        hidden_channels=64
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        lr=1e-3,
        device=device
    )
    
    print("\n=== Model Evaluation ===")
    
    # Evaluate on test set
    mse, mae = evaluate_model(model, test_loader, device=device)
    
    print("\n=== Visualization ===")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.yscale('log')
    
    # Show prediction example
    model.eval()
    with torch.no_grad():
        # Get a sample from test set
        sample_x, sample_y = next(iter(test_loader))
        sample_x, sample_y = sample_x[:1].to(device), sample_y[:1].to(device)  # Take first sample
        
        pred_y = model(sample_x)
        
        # Convert back to numpy and denormalize
        sample_y_np = (sample_y.cpu().numpy() * std_val + mean_val)[0]  # (2, 4, 4)
        pred_y_np = (pred_y.cpu().numpy() * std_val + mean_val)[0]      # (2, 4, 4)
        
        # Convert to complex
        true_complex = sample_y_np[0] + 1j * sample_y_np[1]  # (4, 4)
        pred_complex = pred_y_np[0] + 1j * pred_y_np[1]      # (4, 4)
        
        plt.subplot(1, 2, 2)
        plt.plot(np.abs(true_complex.flatten()), 'b-', label='True', alpha=0.7)
        plt.plot(np.abs(pred_complex.flatten()), 'r--', label='Predicted', alpha=0.7)
        plt.xlabel('Channel Element')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.title('Sample Prediction')
    
    plt.tight_layout()
    plt.savefig('csi_prediction_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== Prediction Quality Analysis ===")
    
    # Analyze prediction quality
    all_mse_per_element = []
    model.eval()
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # MSE per element
            mse_per_element = ((pred - y) ** 2).mean(dim=0)  # Average over batch
            all_mse_per_element.append(mse_per_element.cpu())
    
    # Average MSE per spatial location
    avg_mse_per_element = torch.stack(all_mse_per_element).mean(dim=0)  # (2, 4, 4)
    
    print("Prediction MSE by channel:")
    print(f"Real part MSE: {avg_mse_per_element[0].mean().item():.6f}")
    print(f"Imaginary part MSE: {avg_mse_per_element[1].mean().item():.6f}")
    
    print("\nTraining completed successfully!")
    return model, (train_losses, val_losses), (mse, mae)

if __name__ == "__main__":
    model, losses, metrics = main()
