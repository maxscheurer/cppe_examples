#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import libadcc
import cppe
import numpy as np
import json

from pyscf import gto, scf, lib
from pyscf.solvent import pol_embed
from pyscf.tools import cubegen
from pyscf import solvent

lib.num_threads(32)
thread_pool = libadcc.ThreadPool(32, 32)

# Run SCF in pyscf
mol = gto.M(
    atom="""
C    44.265000    45.233000    45.451000
C    42.946000    45.875000    44.984000
N    42.777000    45.826000    43.543000
C    43.080000    46.961000    42.667000
C    42.233000    48.212000    43.088000
C    42.593000    44.564000    42.956000
C    41.490000    43.850000    43.372000
C    41.101000    42.591000    42.880000
C    41.915000    41.984000    41.876000
C    43.154000    42.687000    41.560000
C    43.414000    43.970000    42.033000
O    44.074000    42.100000    40.757000
C    43.791000    40.854000    40.277000
C    44.607000    40.296000    39.351000
C    44.313000    39.014000    38.606000
O    45.085000    38.573000    37.765000
C    42.983000    38.457000    38.794000
C    42.517000    37.430000    37.964000
C    41.326000    36.779000    38.280000
C    40.447000    37.369000    39.242000
C    40.839000    38.446000    40.040000
C    42.155000    38.995000    39.835000
C    42.538000    40.175000    40.594000
N    41.665000    40.712000    41.489000
H    45.049000    45.851000    45.035000
H    44.516000    44.255000    45.097000
H    44.355000    45.218000    46.563000
H    42.114000    45.433000    45.502000
H    42.861000    46.931000    45.310000
H    44.179000    47.286000    42.641000
H    42.752000    46.568000    41.675000
H    42.611000    48.418000    44.040000
H    41.138000    48.055000    42.941000
H    42.475000    49.165000    42.501000
H    40.832000    44.272000    44.125000
H    40.325000    42.016000    43.331000
H    44.291000    44.409000    41.560000
H    45.508000    40.786000    39.175000
H    43.151000    37.062000    37.180000
H    40.929000    35.954000    37.704000
H    39.378000    37.062000    39.158000
H    40.246000    38.878000    40.859000
    """,
    basis="cc-pvdz",
)

pe_options = cppe.PeOptions()
pe_options.do_diis = True
pe_options.potfile = "nilered_in_blg_1000WAT.pot"
pe = pol_embed.PolEmbed(mol, pe_options)
pe.verbose = 4

lib.num_threads(32)

scfres = solvent.PE(scf.RHF(mol), pe)
scfres.conv_tol = 1e-10
scfres.conv_tol_grad = 1e-8
scfres.verbose = 4
scfres.kernel()

print(adcc.banner())

# Run an adc2 calculation:
state = adcc.adc2(scfres, n_singlets=3, conv_tol=1e-5)

ptss_list = []
ptlr_list = []
# Print results
print()
print(
    "  st  ex.ene. (au)         f     transition dipole moment (au)"
    "        state dip (au)"
    "        ptSS corr (au)         ptLR (au) "
)
for i, val in enumerate(state.excitation_energies):
    tdm_mo = state.transition_dms[i]
    tdm_ao = tdm_mo.to_ao_basis(state.reference_state)
    ρ_tdm_tot = (tdm_ao[0] + tdm_ao[1]).to_ndarray()
    e, _ = scfres._pol_embed.kernel(ρ_tdm_tot, elec_only=True)
    ptlr = 2.0 * e
    ptlr_list.append(ptlr)

    opdm_mo = state.state_diffdms[i]
    opdm_ao = opdm_mo.to_ao_basis(state.reference_state)
    ρdiff_opdm_ao = (opdm_ao[0] + opdm_ao[1]).to_ndarray()
    ptss, _ = scfres._pol_embed.kernel(ρdiff_opdm_ao, elec_only=True)
    ptss_list.append(ptss)

    fmt = "{0:2d}  {1:12.8g} {2:9.3g}   [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    fmt += "   [{6:9.3g}, {7:9.3g}, {8:9.3g}]"
    fmt += "   {9:12.8g} {10:12.8g}"
    print(
        state.kind[0],
        fmt.format(
            i,
            val,
            state.oscillator_strengths[i],
            *state.transition_dipole_moments[i],
            *state.state_dipole_moments[i],
            ptss,
            ptlr
        ),
    )
    # Dump LUNTO and HONTO
    u, s, v = np.linalg.svd(ρ_tdm_tot)
    # LUNTOs
    cubegen.orbital(mol=mol, coeff=u.T[0], outfile="nto_{}_LUNTO.cube".format(i))
    # HONTOs
    cubegen.orbital(mol=mol, coeff=v[0], outfile="nto_{}_HONTO.cube".format(i))

# Print timings summary:
print()
print(state.timer.describe())

print("Number of orbitals:", state.reference_state.n_orbs)
results = {
    "excitation_energies": state.excitation_energies.tolist(),
    "oscillator_strengths": state.oscillator_strengths.tolist(),
    "ptss": ptss_list,
    "ptlr": ptlr_list,
}
with open("results_nile_red_blg.json", "w") as json_file:
    json.dump(results, json_file)
