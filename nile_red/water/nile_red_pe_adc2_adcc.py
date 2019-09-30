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
C    28.793000    32.803000    28.486000
C    29.488000    32.299000    27.320000
N    30.182000    31.015000    27.451000
C    31.237000    31.117000    28.480000
C    30.762000    30.766000    29.900000
C    29.634000    29.842000    27.026000
C    30.269000    28.672000    27.439000
C    29.682000    27.474000    27.145000
C    28.414000    27.350000    26.510000
C    27.792000    28.529000    26.085000
C    28.350000    29.740000    26.403000
O    26.575000    28.569000    25.575000
C    25.919000    27.328000    25.380000
C    24.695000    27.290000    24.781000
C    24.009000    26.059000    24.457000
O    23.015000    26.047000    23.747000
C    24.579000    24.801000    25.082000
C    23.919000    23.564000    24.972000
C    24.464000    22.434000    25.531000
C    25.687000    22.496000    26.159000
C    26.393000    23.708000    26.225000
C    25.828000    24.893000    25.681000
C    26.525000    26.123000    25.886000
N    27.727000    26.169000    26.441000
H    28.024000    32.105000    28.955000
H    28.244000    33.667000    28.216000
H    29.416000    33.206000    29.356000
H    28.764000    32.092000    26.504000
H    30.202000    33.054000    27.009000
H    32.113000    30.624000    28.094000
H    31.531000    32.193000    28.488000
H    29.779000    30.234000    30.024000
H    31.554000    30.090000    30.367000
H    30.744000    31.730000    30.554000
H    31.265000    28.700000    27.955000
H    30.197000    26.528000    27.366000
H    27.784000    30.605000    26.102000
H    24.327000    28.206000    24.327000
H    22.973000    23.598000    24.457000
H    23.982000    21.424000    25.481000
H    26.234000    21.645000    26.532000
H    27.292000    23.851000    26.796000
    """,
    basis="cc-pvdz",
)

pe_options = cppe.PeOptions()
pe_options.do_diis = True
pe_options.potfile = "nilered_in_water.pot"
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
with open("results_nile_red_water.json", "w") as json_file:
    json.dump(results, json_file)
