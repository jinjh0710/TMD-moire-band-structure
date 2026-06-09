'''
Author: Jiahao Jin
This is moire bands of WSe2/MoSe2 heterojunction
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
N = 5       #truncate range
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
a1 = a0 * np.array([1, 0]) #lattice vectors
a2 = Rot(pi/3).dot(a1)
G1 = 4 * pi / sqrt(3) / a0 * np.array([0, 1]) #reciprocal lattice vectors
G2 = Rot(pi/3).dot(G1)
g1 = theta * np.array([G1[1], -G1[0]]) #moire reciprocal lattice vectors
g2 = theta * np.array([G2[1], -G2[0]])
kD = g1[0]/sqrt(3)
Kb = np.array([-kD*sqrt(3)/2, -kD/2])
Kt = np.array([-kD*sqrt(3)/2, kD/2])


#define Lattice
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

Num = 50
KX = list(np.linspace(-Kt[0], 0, Num)) + list(np.linspace(0, Kt[0], Num)) + list(
    np.linspace(Kt[0], Kb[0], Num)) + list(np.linspace(Kb[0], -Kt[0], Num * 2))
KY = list(np.linspace(-Kt[1], 0, Num)) + list(np.linspace(0, Kt[1], Num)) + list(
    np.linspace(Kt[1], Kb[1], Num)) + list(np.linspace(Kb[1], -Kt[1], 2 * Num))

Eigen = np.zeros((len(KX), siteN))

for k in range(len(KX)):
    Eigen[k,:] = SolvHamiltonian(KX[k], KY[k])
Eigen = 1e3 * Eigen # meV

print(theta_d, 'degree VBM: ', max(Eigen[:, -1]))

plt.figure(figsize=(8, 6))
plt.rcParams['axes.linewidth'] = 1
plt.plot(np.linspace(0, 1, len(KX)), Eigen, 'b-', linewidth=2.5)
y_bottom = -60
y_top = 20
plt.ylim(y_bottom, y_top)
plt.xlim(0, 1)
plt.xticks([])
plt.ylabel('E (meV)', fontsize=16)
plt.tick_params(axis='both', labelsize=14)
plt.title(r'$\theta=3^{\circ}$', fontsize=18)
plt.vlines(1/5, y_bottom, y_top, color="k", linewidth=1.5)
plt.vlines(2/5, y_bottom, y_top, color="k", linewidth=1.5)
plt.vlines(3/5, y_bottom, y_top, color="k", linewidth=1.5)
plt.xticks([0, 1/5, 1/5*2, 1/5*3, 1], ['$\kappa^{\'}_{+}$', '$\gamma$', '$\kappa_{-}$', '$\kappa_{+}$', '$\kappa^{\'}_{+}$'], fontsize=15)
plt.tight_layout()
#plt.savefig('data/without_ef/3.0_degree.jpg', dpi=600)
plt.show()
