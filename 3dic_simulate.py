
"""
Static IR-drop analysis of a 3D IC power delivery network (PDN).
 
Model: each tier is an N x N resistive mesh. Tiers are stacked and joined by
TSVs. Supply enters only through C4 bumps on the bottom tier. Each node draws
a DC current. Solve G * V = I (nodal analysis) and report the drop from VDD.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ---------------- parameters ----------------
VDD     = 1.0          # V
N       = 40           # mesh nodes per side, per tier
TIERS   = 2            # number of stacked dies
R_SEG   = 0.05         # ohm, mesh segment resistance (same on every tier)
R_TSV   = 0.05         # ohm, one TSV (tier-to-tier)
R_BUMP  = 0.02         # ohm, one C4 bump to bottom-tier node
TSV_PITCH  = 5         # a TSV every 5 nodes in x and y
BUMP_PITCH = 10        # a bump every 10 nodes in x and y
I_BASE  = [0.2e-3, 0.3e-3]   # A per node, per tier (tier 0 = bottom)
HOTSPOT = {1: (slice(25, 33), slice(8, 16), 3e-3)}  # tier: (rows, cols, A/node)
 
# ---------------- build network ----------------
def idx(t, i, j):
    return t * N * N + i * N + j
 
n_nodes = TIERS * N * N
rows, cols, vals = [], [], []
G_diag = np.zeros(n_nodes)
 
def add_conductance(a, b, g):
    G_diag[a] += g; G_diag[b] += g
    rows.extend([a, b]); cols.extend([b, a]); vals.extend([-g, -g])
 
for t in range(TIERS):
    for i in range(N):
        for j in range(N):
            if j + 1 < N: add_conductance(idx(t, i, j), idx(t, i, j + 1), 1 / R_SEG)
            if i + 1 < N: add_conductance(idx(t, i, j), idx(t, i + 1, j), 1 / R_SEG)
 
for t in range(TIERS - 1):                       # TSVs between tier t and t+1
    for i in range(0, N, TSV_PITCH):
        for j in range(0, N, TSV_PITCH):
            add_conductance(idx(t, i, j), idx(t + 1, i, j), 1 / R_TSV)
 
rhs = np.zeros(n_nodes)
for i in range(0, N, BUMP_PITCH):                # bumps feed the bottom tier
    for j in range(0, N, BUMP_PITCH):
        k = idx(0, i, j)
        G_diag[k] += 1 / R_BUMP
        rhs[k]    += VDD / R_BUMP
 
for t in range(TIERS):                           # load currents
    load = np.full((N, N), I_BASE[t])
    if t in HOTSPOT:
        r, c, amp = HOTSPOT[t]
        load[r, c] = amp
    rhs[t * N * N:(t + 1) * N * N] -= load.ravel()
 
G = sp.csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes)) + sp.diags(G_diag)
 
# ---------------- solve ----------------
V = spla.spsolve(G.tocsc(), rhs)
drop = (VDD - V).reshape(TIERS, N, N)
 
for t in range(TIERS):
    print(f"Tier {t}: max IR drop = {drop[t].max()*1e3:6.2f} mV "
          f"({drop[t].max()/VDD*100:4.2f}% of VDD), mean = {drop[t].mean()*1e3:5.2f} mV")
worst = np.unravel_index(drop.argmax(), drop.shape)
print(f"Worst node: tier {worst[0]}, row {worst[1]}, col {worst[2]}")
 
# ---------------- plot ----------------
fig, axs = plt.subplots(1, TIERS, figsize=(5.5 * TIERS, 4.6))
axs = np.atleast_1d(axs)
vmax = drop.max() * 1e3
for t, ax in enumerate(axs):
    im = ax.imshow(drop[t] * 1e3, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title(f"Tier {t} IR drop (mV)")
fig.colorbar(im, ax=axs.tolist(), shrink=0.85)
fig.savefig("/mnt/user-data/outputs/ir_drop_map.png", dpi=150)
