import psi4

# Set memory
psi4.set_memory("120 GiB")
psi4.set_num_threads(30)
psi4.core.set_output_file("nr_631gd_3roots_vac.dat")

mol = psi4.geometry(
    """
  C   0.27812370431808      1.67799706433198      0.05303450044432
  C   0.45285037034284      1.90525730119212     -1.44800930678034
  N   1.11651102957946      0.80438639287253     -2.13069466714461
  C   0.25344686176467     -0.22440281232970     -2.69043511896325
  C   -0.15209273138513      0.03955743307279     -4.14038489580455
  C   2.48472368371448      0.74440359776252     -2.25453347585288
  C   3.32837158699297      1.69400459940962     -1.60757225981238
  C   4.70363498248368      1.63215484310263     -1.73872425413962
  C   5.32417908319115      0.63143577302088     -2.50381678215105
  C   4.48965804951387     -0.30522723634995     -3.13102587447949
  C   3.10544707901087     -0.26389337931057     -3.02600350310044
  O   5.03761194046719     -1.30070694623165     -3.88564801979747
  C   6.38696309523395     -1.38887650151956     -4.02212638927200
  C   6.90650769076478     -2.39142877944742     -4.76367723162242
  C   8.35585235683613     -2.54734162576218     -4.93914435256588
  O   8.83416170782160     -3.46044106358362     -5.58937445211989
  C   9.22968712662360     -1.52428480766742     -4.27251508472770
  C   10.61753056218062     -1.62487274458797     -4.42541634537020
  C   11.45776189982877     -0.69784058342125     -3.81874336737732
  C   10.91231704402883      0.33957894192036     -3.04989894392275
  C   9.53484656973317      0.44847307335080     -2.89367867687597
  C   8.68182979124663     -0.48266143334276     -3.50512169107743
  C   7.21582322454989     -0.37269719112866     -3.34153452102456
  N   6.70379478644936      0.57247941106098     -2.63135987899233
  H   -0.32105333041565      0.77361952827959      0.24512304832216
  H   -0.23597799167421      2.53307617732096      0.52029071951575
  H   1.25238438959504      1.54841864780001      0.54881996591638
  H   -0.53198612389266      2.04353101042046     -1.92294297574220
  H   1.00351019645540      2.83938846399293     -1.63666753916171
  H   0.74447765874044     -1.20509627033105     -2.59547152957928
  H   -0.64573920719429     -0.28306356418155     -2.05611409908183
  H   0.72843456567176      0.07876802818014     -4.79979223004086
  H   -0.82466026872848     -0.75207205537553     -4.50733885539778
  H   -0.68084988834615      1.00203654505374     -4.22849429771860
  H   2.89820985077741      2.47897636109435     -0.98704679017176
  H   5.34608836041306      2.36218235695722     -1.24015556245550
  H   2.53945957815993     -1.01968591259515     -3.56762747275325
  H   6.25441332443388     -3.12195025246585     -5.24515252758806
  H   11.00476591405813     -2.44823989604193     -5.03017406687360
  H   12.54185219408565     -0.77788203643945     -3.94011704349345
  H   11.57163989232773      1.06881359611591     -2.57031148767225
  H   9.09049939024157      1.25063594580072     -2.30091266349397
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
        "ints_tolerance": 2.5e-11,
    }
)

psi4.set_module_options("ccenergy", {"cachelevel": 0})

psi4.set_module_options("cclambda", {"r_convergence": 1e-3, "cachelevel": 0})
psi4.set_module_options(
    "cceom", {"cachelevel": 0, "r_convergence": 1e-3, "e_convergence": 1e-5}
)

cce, ccwfn = psi4.properties(
    "eom-cc2", properties=["dipole"], return_wfn=True, molecule=mol
)
