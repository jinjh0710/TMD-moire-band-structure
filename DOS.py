'''
Author: Jiahao Jin
This is DOS of moire bands of WSe2/MoSe2 heterojunction
continuum model using plane wave expansion
Reference: Wu et al., 2018, PRL
'''

import numpy as np
from numpy import pi, exp, sqrt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Define constants
theta_d = 3     #degree
N = 6       #truncate range
siteN = (2 * N + 1) ** 2
a0    = 3.32   #angstrom
psi   = -94   #degree
V     = 6.6    #meV
meff  = 0.35    #electron mass
I     = complex(0, 1)

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
a1 = a0 * np.array([1, 0]) # lattice vectors
a2 = Rot(pi/3).dot(a1)
G1 = 4 * pi / sqrt(3) / a0 * np.array([0, 1]) # reciprocal lattice vectors
G2 = Rot(pi/3).dot(G1)
g1 = theta * np.array([G1[1], -G1[0]]) # moire reciprocal lattice vectors
g2 = theta * np.array([G2[1], -G2[0]])
kD = g1[0]/sqrt(3)
Kb = np.array([-kD*sqrt(3)/2, -kD/2])
Kt = np.array([-kD*sqrt(3)/2, kD/2])


# Define Lattice
L = []
invL = np.zeros((2*N+1, 2*N+1), int)
def Lattice(n):
    count = 0
    for i in np.arange(-n, n+1):
        for j in np.arange(-n, n+1):
            L.append([i, j])
            invL[i+n, j+n] = count
            count = count + 1

Lattice(N)
L = np.array(L)

def SolvHamiltonian(kx, ky):
    H = np.zeros((siteN, siteN), dtype=complex)
    for i in np.arange(siteN):
        ix = L[i, 0]
        iy = L[i, 1]

        ax = kx + ix * g1[0] + iy * g2[0]
        ay = ky + ix * g1[1] + iy * g2[1]
        H[i, i] += kin * (ax**2 + ay**2)

        if (ix != N):
            j = invL[ix+1+N, iy+N]
            H[j, i] += V * exp(I * psi)
        if (iy != -N):
            j = invL[ix+N, iy-1+N]
            H[j, i] += V * exp(I * psi)
        if ((ix != -N) and (iy != N)):
            j = invL[ix-1+N, iy+1+N]
            H[j, i] += V*exp(I*psi)
        if (ix != -N):
            j = invL[ix-1 +N, iy +N]
            H[j, i] += V*exp(-I*psi)
        if (iy != N):
            j = invL[ix +N, iy+1 +N]
            H[j, i] += V*exp(-I*psi)
        if ((ix != N) and (iy != -N)):
            j = invL[ix+1 +N, iy-1 +N]
            H[j, i] += V*exp(-I*psi)

    eigenE = np.linalg.eigvalsh(H)
    e = np.sort(eigenE)
    return e

# Generate k lattice grid
def generate_k_points(N_k):
    KX = []
    KY = []
    for kx in np.linspace(-g1[0]/2, g1[0]/2, N_k):
        for ky in np.linspace(-g1[0]/np.sqrt(3), g1[0]/np.sqrt(3), int(2*N_k/sqrt(3))):
            if np.abs(kx)/np.sqrt(3) + np.abs(ky) <= g1[0]/np.sqrt(3):
                KX.append(kx)
                KY.append(ky)
    return KX, KY

N_k = 100  # Density of k grid
KX, KY = generate_k_points(N_k)

eigenE_all = []
report = 0
for kx, ky in zip(KX, KY):
    eigenE_all.extend(SolvHamiltonian(kx, ky))
    report += 1
    print(report)
eigenE_all = np.array(eigenE_all) * 1e3

energy_min = -60
energy_max = np.max(eigenE_all)
energy_range = np.linspace(energy_min, energy_max,400)
bin_width = energy_range[1] - energy_range[0]

# Calculate DOS
dos = np.zeros_like(energy_range)
e_count = 0
for e in eigenE_all:
    index = int((e - energy_min) / bin_width)
    if 0 <= index < len(dos):
        dos[index] += 1
        e_count += 1

# Normalize
dos /= (e_count * bin_width)

# PLot DOS
plt.figure(figsize=(8, 6))
plt.rcParams['axes.linewidth'] = 1
plt.plot(energy_range, dos, color='blue', linewidth=2.5)
plt.xlim(energy_min - 5, energy_max + 5)
plt.title(r'$\theta=3^{\circ}$', fontsize=18)
plt.xlabel('E (meV)', fontsize=16)
plt.ylabel('Density of State',  fontsize=16)
plt.tick_params(axis='x', labelsize=14)
plt.tight_layout()
plt.savefig('data/without_ef/3_degree_DOS.jpg', dpi=600)
plt.show()