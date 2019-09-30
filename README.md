# cppe_examples
Polarizable Embedding (PE) example calculations employing the [CPPE library](https://github.com/maxscheurer/cppe) in various host programs:

- PySCF: CAM-B3LYP/6-31G*, ADC(2)/cc-pVDZ using `adcc` 
- Psi4: EOM-CC2/6-31G*
- Q-Chem: ADC(2)/6-31G*

Chromophore: nile red


## nile_red
- vacuum: optimized structure
- water: PE with water environment (parameters: `nilered_in_water.pot`)
- BLG: PE with beta-lactoglobuling (BLG) environment (parameters: `nilered_in_blg_1000WAT.pot`)
