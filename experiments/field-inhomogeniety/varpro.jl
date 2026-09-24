import VP4Optim as VP
import B0Map as BM
using JLD2
using NPZ

fat_fractions = []
for i in 1:2
    nTE = 16
    t0 = 1.05 + (i-1) * 0.81
    ΔTE = 0.81*2
    TEs = [range(t0, t0 + (nTE-1) * ΔTE, nTE);]
    B0 = 3.0
    precession = :clockwise

    gyro = 42.577e6  # MHz/T
    fat_offset_rad_per_ms = 1.9
    fat_offset_hz = fat_offset_rad_per_ms * 1e3 / (2 * π)
    ppm_fat = [fat_offset_hz / (gyro * B0) * 1e6]
    ampl_fat = [0.729]

    # set up model constructor parameters
    grePar = VP.modpar(BM.GREMultiEchoWF;
        ts = TEs,
        B0 = B0,
        ppm_fat = ppm_fat,
        ampl_fat = ampl_fat,
        precession = precession,
    )

    @load "../../data/results/field-inhomogeniety/data.jld2" data
    @load "../../data/results/field-inhomogeniety/masks.jld2" masks

    data = data[:, :, i:2:end]

    S = masks[1] .| masks[2] .| masks[3] .| masks[4]

    @assert size(data)[1:ndims(S)] == size(S)
    @assert ndims(data) == ndims(S) + 1
    @assert size(data, ndims(data)) == length(TEs)
    @assert eltype(data) <: Complex

    # create an instance of FitPar
    fitpar = BM.fitPar(grePar, data, S)
    fitopt = BM.fitOpt()
    BM.local_fit!(fitpar, fitopt)

    vapro_freq_map = BM.freq_map(fitpar)
    vapro_fat_fraction_map = BM.fat_fraction_map(fitpar, fitopt)
    push!(fat_fractions, vapro_fat_fraction_map)
end

vapro_fat_fraction_map = sum(fat_fractions) ./ length(fat_fractions)

npzwrite("../../data/results/field-inhomogeniety/varpro_fat_fraction_map.npy", vapro_fat_fraction_map)
