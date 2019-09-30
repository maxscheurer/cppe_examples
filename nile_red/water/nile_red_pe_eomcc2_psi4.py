import psi4


def compute_ptss_corrections(ccwfn, nroots):
    ptss = []
    for i in range(nroots):
        ccdmat = ccwfn.variable("CC ROOT {} DA".format(i + 1))
        scfdmat = ccwfn.Da()
        ccdmat.subtract(scfdmat)
        ccdmat.scale(2.0)
        en, op = ccwfn.pe_state.get_pe_contribution(ccdmat, elec_only=True)
        psi4.core.print_out("ptSS {} {:.5f}\n".format(i, en))
        ptss.append(en)
    return ptss


# Set memory
psi4.set_memory("120 GiB")
psi4.set_num_threads(30)
psi4.core.set_output_file("nr_631gd_3roots_water.dat")

mol = psi4.geometry(
    """
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
symmetry c1
no_com
noreorient
"""
)


nroots = 3

psi4.set_options(
    {
        "basis": "6-31G*",
        "freeze_core": "true",
        "roots_per_irrep": [nroots],
        "pe": "true",
        "puream": "true",
        "ints_tolerance": 2.5e-11,
    }
)

psi4.set_module_options("pe", {"potfile": "nilered_in_water.pot", "maxiter": 200})

psi4.set_module_options("ccenergy", {"cachelevel": 0})

psi4.set_module_options("cclambda", {"r_convergence": 1e-3, "cachelevel": 0})
psi4.set_module_options(
    "cceom", {"cachelevel": 0, "r_convergence": 1e-3, "e_convergence": 1e-5}
)

cce, ccwfn = psi4.properties(
    "eom-cc2", properties=["dipole"], return_wfn=True, molecule=mol
)
ptss = compute_ptss_corrections(ccwfn, nroots)
