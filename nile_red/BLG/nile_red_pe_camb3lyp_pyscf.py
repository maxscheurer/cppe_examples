from pyscf import gto
from pyscf.solvent import pol_embed
import cppe
from pyscf import scf
from pyscf import lib
import pyscf.solvent as solvent
import numpy as np
from pyscf.tdscf import TDA
from pyscf.dft import xcfun

np.set_printoptions(linewidth=500)
mol = gto.Mole()
mol.atom = """
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
"""
mol.basis = "6-31G*"
mol.build()
pe_options = cppe.PeOptions()
pe_options.do_diis = True
pe_options.potfile = "nilered_in_blg_1000WAT.pot"
pe = pol_embed.PolEmbed(mol, pe_options)
pe.verbose = 4

lib.num_threads(32)

mf = solvent.PE(scf.RKS(mol), pe)
mf.xc = "camb3lyp"
mf._numint.libxc = xcfun
mf.conv_tol = 1e-8
mf.verbose = 4
mf.kernel()

print(mf._pol_embed.cppe_state.summary_string)

td = TDA(mf)
td.verbose = 5
td.conv_tol = 1e-7
td.triplet = False
td.singlet = True
td.nstates = 3
evals, xy = td.kernel()
td.analyze()
