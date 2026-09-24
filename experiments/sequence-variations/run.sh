#!/bin/bash

for scan in fa10.h5 $(ls ../../data/raw/sequence-variations | grep -v '^fa10\.h5$'); do
    julia --project=../../env/julia -t 4 reco.jl $scan
done

conda run -n chem-comp-mapping python analysis.py
