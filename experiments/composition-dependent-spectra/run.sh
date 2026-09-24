#!/bin/bash

julia +1.9.4 --project=../../env/julia -t 4 reco.jl
conda run -n chem-comp-mapping python analysis.py
