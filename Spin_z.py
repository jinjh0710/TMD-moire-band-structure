'''
Author: Jiahao Jin
This is spin_z of WSe2/MoSe2 heterojunction considering Rashba SOC and Zeeman effect
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
hz = 0.001     #Zeeman term
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


def calculate_spin_z(kx, ky, band_index):
    eigenE, eigenV = SolvHamiltonian(kx, ky)
    vector = eigenV[:, -band_index]
    sigma_z = np.sum(np.abs(vector[0::2]) ** 2 - np.abs(vector[1::2]) ** 2)
    return sigma_z


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
    spin_z = np.zeros(len(KX), dtype=complex)

    for i in range(len(KX)):
        spin_z[i] = calculate_spin_z(KX[i], KY[i], band_index)
        print(i)

    print('max:', max(spin_z))
    print('min:', min(spin_z))
    unique_kx = np.unique(KX)
    unique_ky = np.unique(KY)
    n_kx = len(unique_kx)
    n_ky = len(unique_ky)

    X, Y = np.meshgrid(unique_kx, unique_ky)
    Z = np.full((n_ky, n_kx), np.nan)

    kx_to_idx = {kx: i for i, kx in enumerate(unique_kx)}
    ky_to_idx = {ky: i for i, ky in enumerate(unique_ky)}

    for kx, ky, bc in zip(KX, KY, spin_z):
        i = kx_to_idx[kx]
        j = ky_to_idx[ky]
        Z[j, i] = np.real(bc)

    plt.figure(figsize=(10, 8))

    plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto', vmin=-1, vmax=1)
    #plt.pcolormesh(X, Y, Z, cmap='viridis', shading='auto')
    plt.colorbar()

    vertices = np.array([
        [g1[0] / 2, g1[0] / (2 * np.sqrt(3))],
        [0, g1[0] / np.sqrt(3)],
        [-g1[0] / 2, g1[0] / (2 * np.sqrt(3))],
        [-g1[0] / 2, -g1[0] / (2 * np.sqrt(3))],
        [0, -g1[0] / np.sqrt(3)],
        [g1[0] / 2, -g1[0] / (2 * np.sqrt(3))],
        [g1[0] / 2, g1[0] / (2 * np.sqrt(3))]
    ])
    plt.plot(vertices[:, 0], vertices[:, 1], 'k-', linewidth=2)

    plt.gca().set_aspect('equal')
    plt.axis('off')
    #plt.savefig(f'./data/spin/hz_{1000*hz}meV_band{band_index}_spinz.jpg', dpi=600)
    plt.show()


eigenE, eigenV = SolvHamiltonian(0, 0)
vector = eigenV[:, -1]
sigma_z = np.sum(np.abs(vector[0::2]) ** 2 - np.abs(vector[1::2]) ** 2)
print(sigma_z)
print(vector)

n = 2  # Density of K
band_index = 2 # Index of band
spin = plot_spin_texture(band_index, n)