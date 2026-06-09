'''
Author: Jiahao Jin
This is Berry curvature and Chern number of WSe2/MoSe2 heterojunction considering Rashba SOC and Zeeman effect
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
hz = 0.003     #Zeeman term
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


def wilson_loop_berry_curvature(kx, ky, band_index, delta):
    # Wilson Loop method
    eigenE, eigenV = SolvHamiltonian(kx, ky)
    psi0 = eigenV[:, -band_index]

    eigenE, eigenV = SolvHamiltonian(kx + delta, ky)
    psi_kx = eigenV[:, -band_index]

    eigenE, eigenV = SolvHamiltonian(kx + delta, ky + delta)
    psi_kxky = eigenV[:, -band_index]

    eigenE, eigenV = SolvHamiltonian(kx, ky + delta)
    psi_ky = eigenV[:, -band_index]

    U1 = np.vdot(psi0, psi_kx)
    U2 = np.vdot(psi_kx, psi_kxky)
    U3 = np.vdot(psi_kxky, psi_ky)
    U4 = np.vdot(psi_ky, psi0)

    U = U1 * U2 * U3 * U4
    F = -np.angle(U) / delta / delta
    return F


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


def plot_berry_curvature(band_index, N_k):
    delta = g1[0] / 100 / N_k
    KX, KY = generate_k_points(N_k)
    berry_curvature = np.zeros(len(KX), dtype=complex)

    for i in range(len(KX)):
        berry_curvature[i] = wilson_loop_berry_curvature(KX[i], KY[i], band_index, delta)
        print(i)

    unique_kx = np.unique(KX)
    unique_ky = np.unique(KY)
    n_kx = len(unique_kx)
    n_ky = len(unique_ky)

    X, Y = np.meshgrid(unique_kx, unique_ky)
    Z = np.full((n_ky, n_kx), np.nan)

    kx_to_idx = {kx: i for i, kx in enumerate(unique_kx)}
    ky_to_idx = {ky: i for i, ky in enumerate(unique_ky)}

    for kx, ky, bc in zip(KX, KY, berry_curvature):
        i = kx_to_idx[kx]
        j = ky_to_idx[ky]
        Z[j, i] = np.real(bc)

    plt.figure(figsize=(10, 8))

    plt.pcolormesh(X, Y, Z, cmap='viridis_r', shading='auto')
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
    #plt.savefig(f'./data/band_SOC/{theta_d}_degree_band{band_index}.jpg', dpi=600)
    plt.show()

    '''
    #3D plot band
    max_ky = g1[0] / np.sqrt(3)
    max_kx = g1[0] / 2
    kx_vals = np.linspace(-max_kx, max_kx, 40)
    ky_vals = np.linspace(-max_ky, max_ky, int(2 * 40 / np.sqrt(3)))
    kx_grid, ky_grid = np.meshgrid(kx_vals, ky_vals, indexing='ij')

    bandvalue = np.zeros((len(kx_vals), len(ky_vals)))
    for i in range(len(kx_vals)):
        for j in range(len(ky_vals)):
            eigenE, eigenV = SolvHamiltonian(kx_grid[i, j], ky_grid[i, j])
            bandvalue[i, j] = eigenE[-band_index]
    bandvalue = 1e3 * bandvalue

    fig3D = plt.figure(figsize=(10, 8))
    ax = fig3D.add_subplot(111, projection='3d')
    surf = ax.plot_surface(kx_grid, ky_grid, bandvalue, cmap='viridis', rstride=1, cstride=1, linewidth=0, antialiased=False)
    fig3D.colorbar(surf, ax=ax, shrink=0.5, aspect=10, location='left')
    ax.set_xlabel('$k_x$', fontsize=18)
    ax.set_ylabel('$k_y$', fontsize=18)
    ax.set_zlabel('Energy (meV)', fontsize=18)
    #ax.set_title('Band', fontsize=20)
    # View
    ax.view_init(elev=30, azim=45)
    fig3D.subplots_adjust(left=0.45, right=0.55)
    plt.tight_layout()
    plt.savefig(f'./data/band_SOC/{theta_d}_degree_band{band_index}_3D.jpg', dpi=600)
    plt.show()
    '''

    # Calculate Chern number
    FBZ_area = (np.sqrt(3) / 2) * (g1[0] ** 2)
    dk_area = FBZ_area / len(KX)
    chern_number = np.sum(np.real(berry_curvature)) * dk_area / (2 * pi)

    print(f'Chern number for band {band_index}: {chern_number}')

    #with open(f'./data/band_SOC/{theta_d}_degree_band{band_index}.txt', 'w') as file:
    #    file.write(str(chern_number))

    return np.real(berry_curvature)

n = 50  # Density of K
band_index = 3 # Index of band
berry_curvature = plot_berry_curvature(band_index, n)