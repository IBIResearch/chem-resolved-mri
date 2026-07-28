#!/bin/bash

for ratio in 0.125 0.25 0.5 0.04 0.6 0.8 1.0; do
    julia --project=../../env/julia -t 4 reco.jl $ratio
done

conda run -n chem-comp-mapping python analysis.py
