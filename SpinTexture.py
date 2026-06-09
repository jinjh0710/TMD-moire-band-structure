'''
Author: Jiahao Jin
This is spin texture of WSe2/MoSe2 heterojunction considering Rashba SOC and Zeeman effect
continuum model using plane wave expansion
Reference: Wu et al., 2018, PRL
'''
import numpy as np
from numpy import pi, exp, sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define constants
theta_d = 2  # degree
N = 4  # truncate range
siteN = (2 * N + 1) ** 2
a0 = 3.32  # angstrom
psi = -94  # degree
V = 6.6  # meV
meff = 0.35  # electron mass
alpha = 0.5e-10    #Rashba term
hz = 0.002     #Zeeman term
I = complex(0, 1)

e = 1.60218e-19
V = V * 1e-3
psi = psi * pi / 180
hbar = 6.62607e-34 / 2 / pi
me = 9.10938e-31
m_eff = meff * me
kin = -hbar ** 2 / 2 / m_eff / e


def Rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


theta = theta_d * pi / 180
a0 = a0 * 1e-10
a1 = a0 * np.array([1, 0])  # lattice vectors
a2 = Rot(pi / 3).dot(a1)
G1 = 4 * pi / sqrt(3) / a0 * np.array([0, 1])  # reciprocal lattice vectors
G2 = Rot(pi / 3).dot(G1)
g1 = theta * np.array([G1[1], -G1[0]])  # moire reciprocal lattice vectors
g2 = theta * np.array([G2[1], -G2[0]])
kD = g1[0] / sqrt(3)
Kb = np.array([-kD * sqrt(3) / 2, -kD / 2])
Kt = np.array([-kD * sqrt(3) / 2, kD / 2])

# define Lattice
L = []
invL = np.zeros((2 * N + 1, 2 * N + 1), int)


def Lattice(n):
    count = 0
    for i in np.arange(-n, n + 1):
        for j in np.arange(-n, n + 1):
            L.append([i, j])
            invL[i + n, j + n] = count
            count = count + 1

Lattice(N)
L = np.array(L)

def SolvHamiltonian(kx, ky):
    H = np.zeros((2*siteN, 2*siteN), dtype=complex)
    for i in np.arange(siteN):
        ix = L[i, 0]
        iy = L[i, 1]

        ax = kx + ix * g1[0] + iy * g2[0]
        ay = ky + ix * g1[1] + iy * g2[1]
        H[2*i, 2*i] += kin * (ax ** 2 + ay ** 2) + hz
        H[2*i+1, 2*i+1] += kin * (ax ** 2 + ay ** 2) - hz

        H[2 * i, 2 * i + 1] += alpha * (ay + I * ax)
        H[2 * i + 1, 2 * i] += alpha * (ay - I * ax)

        if (ix != N):
            j = invL[ix + 1 + N, iy + N]
            H[2 * j, 2 * i] += V * exp(I * psi)
            H[2 * j + 1, 2 * i + 1] += V * exp(I * psi)
        if (iy != -N):
            j = invL[ix + N, iy - 1 + N]
            H[2 * j, 2 * i] += V * exp(I * psi)
            H[2 * j + 1, 2 * i + 1] += V * exp(I * psi)
        if ((ix != -N) and (iy != N)):
            j = invL[ix - 1 + N, iy + 1 + N]
            H[2 * j, 2 * i] += V * exp(I * psi)
            H[2 * j + 1, 2 * i + 1] += V * exp(I * psi)
        if (ix != -N):
            j = invL[ix - 1 + N, iy + N]
            H[2 * j, 2 * i] += V * exp(-I * psi)
            H[2 * j + 1, 2 * i + 1] += V * exp(-I * psi)
        if (iy != N):
            j = invL[ix + N, iy + 1 + N]
            H[2 * j, 2 * i] += V * exp(-I * psi)
            H[2 * j + 1, 2 * i + 1] += V * exp(-I * psi)
        if ((ix != N) and (iy != -N)):
            j = invL[ix + 1 + N, iy - 1 + N]
            H[2 * j, 2 * i] += V * exp(-I * psi)
            H[2 * j + 1, 2 * i + 1] += V * exp(-I * psi)

    eigenE, eigenV = eigh(H)
    idx = eigenE.argsort()
    eigenE = eigenE[idx]
    eigenV = eigenV[:, idx]
    return eigenE, eigenV


def calculate_spin_components(kx, ky, band_index):
    eigenE, eigenV = SolvHamiltonian(kx, ky)
    vector = eigenV[:, -band_index]

    # Calculate spin expectation values
    sigma_x = np.sum(vector[0::2].conj() * vector[1::2] + vector[1::2].conj() * vector[0::2])
    sigma_y = np.sum(I * (vector[1::2].conj() * vector[0::2] - vector[0::2].conj() * vector[1::2]))
    return np.real(sigma_x), np.real(sigma_y)


def generate_k_points(N_k):
    KX = []
    KY = []
    max_ky = g1[0] / np.sqrt(3)
    max_kx = g1[0] / 2

    kx_values = np.linspace(-max_kx, max_kx, N_k)
    ky_values = np.linspace(-max_ky, max_ky, int(2 * N_k / np.sqrt(3)))

    for kx in kx_values:
        for ky in ky_values:
            if np.abs(kx) / np.sqrt(3) + np.abs(ky) <= max_ky:
                KX.append(kx)
                KY.append(ky)
    return np.array(KX), np.array(KY)



def plot_spin_texture(band_index, N_k):
    KX, KY = generate_k_points(N_k)
    spin_x = np.zeros(len(KX))
    spin_y = np.zeros(len(KX))

    for i in range(len(KX)):
        sx, sy = calculate_spin_components(KX[i], KY[i], band_index)
        spin_x[i] = sx
        spin_y[i] = sy
        print(f"Progress: {i + 1}/{len(KX)}")

    unique_kx = np.unique(KX)
    unique_ky = np.unique(KY)
    n_kx = len(unique_kx)
    n_ky = len(unique_ky)

    X, Y = np.meshgrid(unique_kx, unique_ky)
    Z_x = np.full((n_ky, n_kx), np.nan)
    Z_y = np.full((n_ky, n_kx), np.nan)

    kx_to_idx = {kx: i for i, kx in enumerate(unique_kx)}
    ky_to_idx = {ky: i for i, ky in enumerate(unique_ky)}

    for kx, ky, sx, sy in zip(KX, KY, spin_x, spin_y):
        i = kx_to_idx[kx]
        j = ky_to_idx[ky]
        Z_x[j, i] = sx
        Z_y[j, i] = sy


    # Normalize the arrows for better visualization
    norm = np.sqrt(Z_x ** 2 + Z_y ** 2)
    norm[norm == 0] = 1  # Avoid division by zero
    Zx_norm = Z_x / norm
    Zy_norm = Z_y / norm

    # Plot spin texture in x-y plane
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, Zx_norm, Zy_norm, norm, cmap='viridis', scale=22, width=0.004, headwidth=6, headlength=8)
    cbar = plt.colorbar(shrink=0.85)
    cbar.ax.set_ylabel('Spin Magnitude', fontsize=16)

    # Plot the first Brillouin zone boundary
    vertices = np.array([
        [g1[0] / 2, g1[0] / (2 * np.sqrt(3))],
        [0, g1[0] / np.sqrt(3)],
        [-g1[0] / 2, g1[0] / (2 * np.sqrt(3))],
        [-g1[0] / 2, -g1[0] / (2 * np.sqrt(3))],
        [0, -g1[0] / np.sqrt(3)],
        [g1[0] / 2, -g1[0] / (2 * np.sqrt(3))],
        [g1[0] / 2, g1[0] / (2 * np.sqrt(3))]
    ])
    plt.plot(vertices[:, 0], vertices[:, 1], color='0.6', linestyle='--', linewidth=2)
    plt.title(f'Bottom Flat Band (h$_z$ = {1000 * hz} meV)', fontsize=20)
    plt.xlabel('$k_x$')
    plt.ylabel('$k_y$')
    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.tight_layout()
    #plt.savefig(f'./data/spin/ST_hz_{1000*hz}meV_band{band_index}.png', dpi=300)
    plt.show()



n = 21  # Density of K
band_index = 1 # Index of band
spin = plot_spin_texture(band_index, n)


