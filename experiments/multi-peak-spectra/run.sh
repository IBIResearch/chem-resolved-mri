#!/bin/bash

# julia --project=../../env/julia -t 4 reco.jl
conda run -n chem-comp-mapping python analysis.py
