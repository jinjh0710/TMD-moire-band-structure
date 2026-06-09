# MoSe2/WSe2 Moiré Superlattice Simulation Suite

Python codes for simulating the electronic properties of MoSe2/WSe2 moiré superlattices under external fields, including Rashba spin-orbit coupling (SOC) and Zeeman effects.

## Features

- Band structure calculation using TMD continuum model
- Density of states (DOS) analysis
- Berry curvature and Chern number calculation via Wilson loop method
- Flat band width vs. twist angle analysis
- Tight-binding Hubbard model description of flat bands
- Rashba SOC and Zeeman term implementation
- Spin polarization ⟨σ_z⟩ calculation
- In-plane spin texture (⟨σ_x⟩, ⟨σ_y⟩) visualization

## Code Description

| File | Description |
|------|-------------|
| `Band.py` | Band structure from TMD continuum model |
| `DOS.py` | Density of states |
| `Topology.py` | Berry curvature and Chern numbers (Wilson loop) |
| `Band_TightBinding.py` | Hubbard model for flat bands |
| `BandWidth.py` | Flat band width vs. twist angle |
| `Band_SOC.py` | Band structure with Rashba SOC + Zeeman |
| `Topology_SOC.py` | Topology with SOC + Zeeman terms |
| `Spin_z.py` | Expectation value ⟨σ_z⟩ |
| `SpinTexture.py` | In-plane spin texture ⟨σ_x, σ_y⟩ |

## Requirements
