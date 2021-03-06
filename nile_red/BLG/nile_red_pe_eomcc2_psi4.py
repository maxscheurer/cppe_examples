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
psi4.core.set_output_file("nr_631gd_3roots_blg.dat")

mol = psi4.geometry(
    """
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
        "ints_tolerance": 2.5e-11,
        "puream": "true",
    }
)

psi4.set_module_options("pe", {"potfile": "nilered_in_blg_1000WAT.pot", "maxiter": 200})

psi4.set_module_options("ccenergy", {"cachelevel": 0})

psi4.set_module_options("cclambda", {"r_convergence": 1e-3, "cachelevel": 0})
psi4.set_module_options(
    "cceom", {"cachelevel": 0, "r_convergence": 1e-3, "e_convergence": 1e-5}
)

cce, ccwfn = psi4.properties(
    "eom-cc2", properties=["dipole"], return_wfn=True, molecule=mol
)
ptss = compute_ptss_corrections(ccwfn, nroots)
