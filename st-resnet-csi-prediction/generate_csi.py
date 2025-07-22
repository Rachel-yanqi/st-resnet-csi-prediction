
import numpy as np

def exponential_corr_matrix(N, decay=0.5):
    return np.array([[decay**abs(i-j) for j in range(N)] for i in range(N)])

def generate_spatial_temporal_correlated_csi(num_symbols=1000, num_users=4, k_factor_dB=6.0,
                                              rho=0.95, R_tx=None, R_rx=None):
    k_linear = 10**(k_factor_dB / 10)
    alpha = np.sqrt(k_linear / (k_linear + 1))
    beta = np.sqrt(1 / (k_factor_dB + 1))

    if R_tx is None:
        R_tx = np.eye(num_users)
    if R_rx is None:
        R_rx = np.eye(num_users)

    sqrt_R_tx = np.linalg.cholesky(R_tx)
    sqrt_R_rx = np.linalg.cholesky(R_rx)

    h_los = np.ones((num_users, num_users), dtype=np.complex64)

    H = np.zeros((num_symbols, num_users, num_users), dtype=np.complex64)
    W_prev = (np.random.randn(num_users, num_users) + 1j * np.random.randn(num_users, num_users)) / np.sqrt(2)

    for t in range(num_symbols):
        noise = (np.random.randn(num_users, num_users) + 1j * np.random.randn(num_users, num_users)) / np.sqrt(2)
        W_curr = rho * W_prev + np.sqrt(1 - rho**2) * noise
        W_prev = W_curr

        h_nlos = sqrt_R_rx @ W_curr @ sqrt_R_tx
        h_t = alpha * h_los + beta * h_nlos
        H[t] = h_t

    return H
