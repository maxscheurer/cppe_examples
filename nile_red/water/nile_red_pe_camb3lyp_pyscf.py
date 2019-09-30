from pyscf import gto
from pyscf.solvent import pol_embed
import cppe
from pyscf import scf
from pyscf import lib
from pyscf.dft import xcfun
import pyscf.solvent as solvent
import numpy as np
from pyscf.tdscf import TDA

np.set_printoptions(linewidth=500)
mol = gto.Mole()
mol.atom = """
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
"""
mol.basis = "6-31G*"
mol.build()
pe_options = cppe.PeOptions()
pe_options.do_diis = True
pe_options.potfile = "nilered_in_water.pot"
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
