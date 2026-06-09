'''
Author: Jiahao Jin
This is moire bands of WSe2/MoSe2 heterojunction and calculate the bandwidth of the first flat band
continuum model using plane wave expansion
Reference: Wu et al., 2018, PRL
'''

import numpy as np
from numpy import pi, exp, sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define constants
theta_d_list = np.linspace(0.5, 3, 100)
N = 6  # truncate range
siteN = (2 * N + 1) ** 2
a0 = 3.32  # angstrom
psi = -94  # degree
V = 6.6  # meV
meff = 0.35  # electron mass
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

a0_scaled = a0 * 1e-10
a1 = a0_scaled * np.array([1, 0])  # lattice vectors
a2 = Rot(pi / 3).dot(a1)
G1 = 4 * pi / sqrt(3) / a0_scaled * np.array([0, 1])  # reciprocal lattice vectors
G2 = Rot(pi / 3).dot(G1)

def setup_system(theta_d):
    theta = theta_d * pi / 180
    g1 = theta * np.array([G1[1], -G1[0]])  # moire reciprocal lattice vectors
    g2 = theta * np.array([G2[1], -G2[0]])
    kD = g1[0] / sqrt(3)
    Kb = np.array([-kD * sqrt(3) / 2, -kD / 2])
    Kt = np.array([-kD * sqrt(3) / 2, kD / 2])
    Gamma = np.array([0.0, 0.0])

    return Gamma, Kt, g1, g2


# Define Lattice
def setup_lattice(n):
    L = []
    invL = np.zeros((2 * n + 1, 2 * n + 1), int)
    count = 0
    for i in np.arange(-n, n + 1):
        for j in np.arange(-n, n + 1):
            L.append([i, j])
            invL[i + n, j + n] = count
            count += 1
    return np.array(L), invL

L, invL = setup_lattice(N)

def SolvHamiltonian(kx, ky, g1, g2):
    H = np.zeros((siteN, siteN), dtype=complex)
    for i in np.arange(siteN):
        ix = L[i, 0]
        iy = L[i, 1]

        ax = kx + ix * g1[0] + iy * g2[0]
        ay = ky + ix * g1[1] + iy * g2[1]
        H[i, i] += kin * (ax ** 2 + ay ** 2)

        if (ix != N):
            j = invL[ix + 1 + N, iy + N]
            H[j, i] += V * exp(I * psi)
        if (iy != -N):
            j = invL[ix + N, iy - 1 + N]
            H[j, i] += V * exp(I * psi)
        if ((ix != -N) and (iy != N)):
            j = invL[ix - 1 + N, iy + 1 + N]
            H[j, i] += V * exp(I * psi)
        if (ix != -N):
            j = invL[ix - 1 + N, iy + N]
            H[j, i] += V * exp(-I * psi)
        if (iy != N):
            j = invL[ix + N, iy + 1 + N]
            H[j, i] += V * exp(-I * psi)
        if ((ix != N) and (iy != -N)):
            j = invL[ix + 1 + N, iy - 1 + N]
            H[j, i] += V * exp(-I * psi)

    eigenE = np.linalg.eigvalsh(H)
    return np.sort(eigenE)


# Calculate energy differences for different angles
energy_differences = []

for theta_d in theta_d_list:
    Gamma, Kt, g1, g2 = setup_system(theta_d)

    # Calculate energies at Gamma and Kt points
    E_Gamma = SolvHamiltonian(Gamma[0], Gamma[1], g1, g2)
    E_Kt = SolvHamiltonian(Kt[0], Kt[1], g1, g2)

    # Energy difference between first band at Gamma and Kt points
    delta_E = E_Gamma[-1] - E_Kt[-1]
    energy_differences.append(delta_E * 1e3)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(theta_d_list, energy_differences, 'b-', linewidth=2.5, markersize=8)
plt.xlabel(r'Twist Angle $(^\circ)$', fontsize=18)
plt.ylabel('Energy (meV)', fontsize=18)
plt.title('Bandwidth of the Flat Band', fontsize=20)
plt.tight_layout()
plt.savefig(f'data/without_ef/bandwidth.jpg', dpi=300)
plt.show()
