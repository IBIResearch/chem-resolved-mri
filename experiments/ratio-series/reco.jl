### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ bdaba444-746d-4c14-a707-432fb3d9a03f
begin
	using Pkg
	Pkg.activate("../../env/julia")
end

# ╔═╡ 355a9914-dbdb-49d0-9186-711c7ac1e500
begin
	using Revise
	using MRIFiles, MRIBase, HDF5, ChemResolvedMRI
	using NFFT, ImageSegmentation, Images, Statistics
	using ColorSchemes, Colors, StatsPlots, GLM, SavitzkyGolay
	using DataFrames, JSON, NPZ, Plots
	using RegularizedLeastSquares, MRIReco, LaTeXStrings
	using Interpolations, FFTW
end

# ╔═╡ 659762fb-ba8e-4ae4-b1f7-6c209ac717b2
html"""
<style>
main {
    max-width: 95%;  /* Default is ~65%, increasing to 95% */
}
</style>
"""

# ╔═╡ bd9c7fc3-7a86-4d8e-8e39-bc6763168cdd
begin
	DATA_FOLDER = "../../data"
	EXPERIMENT_NAME = "ratio-series"
end

# ╔═╡ 5bd490cd-316f-40cc-8354-e25b8659c6af
md"""
# Data Loading
"""

# ╔═╡ 23a171d3-6c7b-4fe6-9a3c-824c969fdd54
begin
	n_species = 2
	expected_acetone_ratio_str = get_arg("1.0")
	measurements = Dict(
		"0.04"=>["ratio_series_1.h5"],
		"0.125"=>["ratio_series_2.h5"],
		"0.25"=>["ratio_series_3.h5"],
		"0.5"=>["ratio_series_4.h5"],
		"0.6"=>["ratio_series_5.h5"],
		"0.8"=>["ratio_series_6.h5"],
		"1.0"=>["ratio_series_7.h5"]
	)[expected_acetone_ratio_str]
	expected_acetone_ratio = parse(Float64, expected_acetone_ratio_str)

	RESULTS_FOLDER = joinpath(DATA_FOLDER, "results", EXPERIMENT_NAME, "acetone-$expected_acetone_ratio")
	mkpath(RESULTS_FOLDER)
end

# ╔═╡ 775edc9a-aae5-4e6f-b465-f54f07e20923
begin
	acqs = []
	echo_times = []
	echo_time2meas_n_echo_n = Dict{Float64, Tuple{Int, Int}}()
	for (i, meas_file_name) in enumerate(measurements)
		mge_meas_path = joinpath(DATA_FOLDER, "raw", meas_file_name)

		raw = RawAcquisitionData(ISMRMRDFile(mge_meas_path))
		acq = AcquisitionData(raw)
	    push!(acqs, acq)

		h5open(mge_meas_path, "r") do file
			push!(echo_times, read(file["echo_times"]))
		end

		for (echo_n, t) in enumerate(echo_times[end])
			if haskey(echo_time2meas_n_echo_n, t)
				@warn "bijective mapping is not possible: there is entry $(echo_time2meas_n_echo_n[t]) for $t, skipping $((length(acqs), echo_n))"
				continue
			end
			echo_time2meas_n_echo_n[t] = (length(acqs), echo_n)
		end

	end
	all_echo_times = copy(echo_times)

	echo_times_vec = vcat(echo_times...)
	echo_times_sorted_index = sortperm(echo_times_vec);

	invert_readout_for_odd_echos!(acqs)
	shift_ky!(acqs)
end

# ╔═╡ 444ac30f-bf7a-47b2-a59c-a1c6eec81cdc
begin
	nx, ny = acqs[1].encodingSize[1], acqs[1].encodingSize[2]
	nc = 2
	n_points_per_profile = nx
	n_profiles = ny
end

# ╔═╡ 48499038-a513-4871-96a9-f359f6732122
md"""
# Per-Echo Reconstructions
"""

# ╔═╡ a18153a4-f78c-4ac7-b605-53602cf5f96e
begin
	D = 2
	J = nx*ny
	N = (nx, ny)
	k  = MRIBase.cartesian2dNodes(Float32, ny, nx; kmin=(-0.5, -0.5), kmax=(0.5, 0.5))
	nfft_plan = NFFTPlan(k, N; m=4, σ=2)
end

# ╔═╡ e6d8d705-5558-4dfd-a999-f140984e0398
begin
	recos = []
	for acq in acqs
	    # we take only the first coil
	    kspace = kDataCart(acq)[:, :, 1, 1, :, 1]
	    reco = zeros(eltype(kspace), size(kspace)...)
	    for i=1:size(kspace, 3)
	        reco[:, :, i] .= adjoint(nfft_plan) * vec(kspace[:, :, i])
	    end
	    reco ./= prod(N)
	    push!(recos, reco)
	end
	acq = acqs[1]
	direct_reco = recos[1]

	nothing
end

# ╔═╡ d5749a21-3b8b-4141-aee8-dd25cd3f2c40
let
	n_echos = length(echo_times[1])
	_echo_plots = [
	    heatmap(
	        abs.(direct_reco)[div(nx, 4):end-div(nx, 4), :, i],
	        c = :grays,
	        aspect_ratio=:equal,
	        showaxis=false, colorbar=false,
	        title="Echo #$i, $(round(t, digits=2))ms", clim=(0, maximum(abs.(direct_reco))),
	        titlefontcolor=:white
	    )
	    for (i, t)=zip(1:n_echos, echo_times[1])
	]

	p = plot(
	    _echo_plots...,
	    layout=(4, 8),
	    plot_title="\nReconstructions for Each Echo",
	    size=(1920, 1080),
	    dpi=300,
	    background_color=:black,
	    plot_titlefontcolor=:white,
	    plot_titlevspan=0.08,
	    plot_titlefontvalign=:bottom,
	    framestyle=:box,
	)
	savefig(p, joinpath(RESULTS_FOLDER, "echo-recons-magnitude.png"))
	p
end

# ╔═╡ c507bfdd-7e9c-4d53-8b0c-6c31b28f66ce
let
	n_echos = length(echo_times[1])
	_echo_plots = [
	    heatmap(
	        angle.(direct_reco)[div(nx, 4):end-div(nx, 4), :, i] .* (abs.(direct_reco)[div(nx, 4):end-div(nx, 4), :, 4] .> 0.05),
	        c = :grays,
	        aspect_ratio=:equal,
	        showaxis=false, colorbar=false,
	        title="Echo #$i, $(round(t, digits=2))ms", clim=(-π, π),
	    )
	    for (i, t)=zip(1:n_echos, echo_times[1])
	]

	p = plot(
	    _echo_plots...,
	    layout=(4, 8),
	    plot_title="\n Phases for Each Echo",
	    size=(1920, 1080),
	    dpi=300,
	    plot_titlevspan=0.08,
	    plot_titlefontvalign=:bottom,
	    framestyle=:box,
	)
	savefig(p, joinpath(RESULTS_FOLDER, "echo-recons-phase.png"))
	p
end

# ╔═╡ 87e61d59-a098-476d-bdec-b88864f1a9f3
md"""
# Segmentation
"""

# ╔═╡ 714619d9-e6d7-4d91-ad91-e616da9b2bb1
seeds = Vector{Tuple{CartesianIndex{2}, Int}}([
    (CartesianIndex(80, 40), 1),
    (CartesianIndex(70, 30), 2),
    (CartesianIndex(1, 1), 3),
])

# ╔═╡ e09ee2cf-ec4c-4c67-89b9-e263751a3a14
begin
	magnitude_image = mean([abs.(adjoint(nfft_plan) * vec(mean(kDataCart(acq)[:, :, 1, :, i, 1]; dims=3))) for i in 1:5])
	result = map(scaleminmax(Float64, 0.0, 256.0), magnitude_image)
	save(joinpath(RESULTS_FOLDER, "magnitude_image.png"), result)
	segments = seeded_region_growing(magnitude_image, seeds)
end

# ╔═╡ 4d85e386-ce43-4f62-bdfa-81c53218ec66
begin
	masks = []

	local p = heatmap(
	    magnitude_image[div(nx, 4):end-div(nx, 4), :],
	    c = :grays, aspect_ratio=:equal, showaxis=false, colorbar=false
	)

	for label in 1:3
	    mask = labels_map(segments) .== label
	    if label == 1 || label == 2
	        mask = erode(mask)
	    end
	    push!(masks, mask)
	    mask = mask[div(nx, 4):end-div(nx, 4), :]
	    overlay = mask_to_rgb(mask, RGBA(get(ColorSchemes.lajolla100, label/length(seeds)), 0.2))
	    p = plot!(p, overlay)

	    mask_center = seeds[label][1]
	    annotate!(p, mask_center[1]-div(nx, 4), mask_center[2], text("#$label", :black, :center, 10))
	end
	p
end

# ╔═╡ 9b8342d4-8014-4129-8e4f-f10d31249858
md"""
# Forward Model
"""

# ╔═╡ 5ead8775-8a12-4a4e-8e17-f6cf110d0892
md"""
## Frequency estimation
"""

# ╔═╡ 0e7f25ae-887d-459e-a8f2-f1ecd1905768
let
    # --- Extract time and signals ---
    t = echo_times_vec[echo_times_sorted_index]
    y_inner_tube = vec(mean(
        cat(recos...; dims=3)[:, :, echo_times_sorted_index][masks[1], :];
        dims=1
    ))
    y_outer_tube = vec(mean(
        cat(recos...; dims=3)[:, :, echo_times_sorted_index][masks[2], :];
        dims=1
    ))

    # --- Interpolate to uniform grid ---
    Δt = 0.13
    t_uniform = t[1]:Δt:t[end]
    itp_inner = interpolate((t,), y_inner_tube, Gridded(Linear()))
    y_inner_uniform = itp_inner.(t_uniform)
    itp_outer = interpolate((t,), y_outer_tube, Gridded(Linear()))
    y_outer_uniform = itp_outer.(t_uniform)

    # --- FFT ---
    Y_inner = fftshift(fft(y_inner_uniform))
    Y_outer = fftshift(fft(y_outer_uniform))
    fs = 1000 / Δt
    freqs = (-div(length(Y_inner), 2):length(Y_inner)-div(length(Y_inner), 2)-1) .* (fs / length(Y_inner))
    global mag_inner = abs.(Y_inner)
    global mag_outer = abs.(Y_outer)

    # --- Peak finder ---
    function find_peaks(mag, freqs, n_expected)
        idx = sortperm(mag, rev=true)[1:n_expected]
        return freqs[idx], mag[idx]
    end

    peak_freqs_inner, peak_vals_inner = find_peaks(mag_inner[freqs .< -100], freqs[freqs .< -100], 1)
    peak_freqs_outer, peak_vals_outer = find_peaks(mag_outer, freqs, 1)

	@assert length(peak_freqs_inner) == 1
	@assert length(peak_vals_outer) == 1

	global acetone_freq = peak_freqs_inner[1] * 2pi / 1000
	global water_freq = peak_freqs_outer[1] * 2pi / 1000

    # --- Plot ---
    p = plot(
        freqs, mag_inner,
        xlabel = "Frequency (Hz)",
        ylabel = "Magnitude",
        title = "Fourier Spectra (Inner vs Outer Tube)",
        linewidth = 2,
        linecolor = :blue,
        label = "Inner Tube",
        grid = true,
        framestyle = :box
    )

    plot!(
        freqs, mag_outer,
        linewidth = 2,
        linecolor = :green,
        label = "Outer Tube"
    )

    scatter!(peak_freqs_inner, peak_vals_inner,
        color = :red, markerstrokecolor = :black, markersize = 6, label = "")
    scatter!(peak_freqs_outer, peak_vals_outer,
        color = :orange, markerstrokecolor = :black, markersize = 6, label = "")

    for (f, v) in zip(peak_freqs_inner, peak_vals_inner)
        annotate!(f, v, text("$(round(f, digits=2)) Hz", :red, 8, :center))
    end
    for (f, v) in zip(peak_freqs_outer, peak_vals_outer)
        annotate!(f, v, text("$(round(f, digits=2)) Hz", :orange, 8, :center))
    end
    p
end

# ╔═╡ 3be0cb02-2192-4866-8173-a5fff6f5a0bc
md"""
## T2* estimation (run for pure acetone only)
"""

# ╔═╡ 6ea2bffc-6cf4-4f56-aec9-02d0048336e3
function estimate_T2star(mask_n)
	t = echo_times_vec[echo_times_sorted_index]
    y = cat(recos...; dims=3)[:, :, echo_times_sorted_index][masks[mask_n], :]
	y_magn = abs.(y)
	log_y_magn = log.(y_magn)

	T2_star_values = []
	for i=1:size(y, 1)
		df = DataFrame(
			X = t,
			Y = Float64.(log_y_magn[i, :]),
		)
		log_magn_ols = lm(@formula(Y ~ X), df)
		R = -coef(log_magn_ols)[2]
		T2_star = 1/R
		push!(T2_star_values, T2_star)
	end
	return filter_outliers(T2_star_values)
end

# ╔═╡ 8ac278be-d2b7-4ab5-bde2-38fb9c231eb0
begin
	if expected_acetone_ratio_str == "1.0"
		T2_star_values_acetone = estimate_T2star(1)
		T2_star_values_water = estimate_T2star(2)
		pa = histogram(T2_star_values_acetone)
		pw = histogram(T2_star_values_water)
		@info "T2* acetone: $(mean(T2_star_values_acetone))\nT2* water: $(mean(T2_star_values_water))"
		npzwrite(joinpath(RESULTS_FOLDER, "T2_star_values_acetone.npy"), Float64.(T2_star_values_acetone))
		npzwrite(joinpath(RESULTS_FOLDER, "T2_star_values_water.npy"), Float64.(T2_star_values_water))
		plot(
			pw, pa,
			layout=(1, 2)
		)
	end
end

# ╔═╡ 4766f241-ba3d-4ef8-85dc-84116e0ee161
let
	t = echo_times_vec[echo_times_sorted_index]
    y = cat(recos...; dims=3)[:, :, echo_times_sorted_index][masks[2], :]
	y_magn = abs.(y)
	log_y_magn = log.(y_magn)

	global T2_star_values = []
	for i=1:size(y, 1)
		df = DataFrame(
			X = t,
			Y = Float64.(log_y_magn[i, :]),
		)
		log_magn_ols = lm(@formula(Y ~ X), df)
		R = -coef(log_magn_ols)[2]
		T2_star = 1/R
		push!(T2_star_values, T2_star)
	end
end

# ╔═╡ 5051eb11-3251-495e-ace0-3e3679220fea
md"""
## Parametrization
"""

# ╔═╡ cada972b-655a-4519-8e43-d588d5b63a86
begin
	df = convert(Vector{Vector{Float32}}, [
	    [water_freq],
	    [acetone_freq],
	])
	phi0 = convert(Vector{Float32}, [
	    0.0,
	    0.0,
	])
	weights = convert(Vector{Vector{Float32}}, [
		[Parameters.M0_water],
	    [Parameters.M0_acetone],
	])
	# uncomment to run with relaxation
	# relaxation = [
	# 	1/345,
	# 	1/324,
	# ]
	relaxation = nothing

	f = open(RESULTS_FOLDER * "/model_frequencies.json", "w")
	JSON.print(f, Dict("acetone" => acetone_freq, "water" => water_freq), 4)
	close(f)
end

# ╔═╡ a577da49-df44-4468-97b6-6a4fd1207a0f
md"""
# EPSI
"""

# ╔═╡ 04d5f030-0115-4cd2-bcc4-57659d4fcd23
let
	epsi = fftshift(fft(recos[1], 3), 3);

	fs = 1000 / median(echo_times_vec[2:end] .- echo_times_vec[1:end-1])
    freqs = (-div(length(echo_times_vec), 2):length(echo_times_vec)-div(length(echo_times_vec), 2)-1) .* (fs / length(echo_times_vec))

	df = 75
	water_freq_mask = (water_freq / 2pi * 1000 - df) .< freqs .< (water_freq / 2pi * 1000 + df)
	acetone_freq_mask = (acetone_freq / 2pi * 1000 - df) .< freqs .< (acetone_freq / 2pi * 1000 + df)

	data = Dict(
		"acetone_c_values_in_mixture"=>abs.(sum(abs.(epsi[:, :, acetone_freq_mask]), dims=3)[:, :, 1][erode(masks[1])]),
		"water_c_values_in_mixture"=>abs.(sum(abs.(epsi[:, :, water_freq_mask]), dims=3)[:, :, 1][erode(masks[1])]),
		"acetone_c_values_in_water"=>abs.(sum(abs.(epsi[:, :, acetone_freq_mask]), dims=3)[:, :, 1][erode(masks[2])]),
		"water_c_values_in_water"=>abs.(sum(abs.(epsi[:, :, water_freq_mask]), dims=3)[:, :, 1][erode(masks[2])]),
	)
	open(joinpath(RESULTS_FOLDER, "abs_concentrations_of_species_epsi.json"), "w") do f
	  JSON.print(f, data, 4)
	end
end

# ╔═╡ 6d6d0e0d-7aad-49c7-ad45-298928745d1a
md"""
# Chemically Resolved Reconstruction
"""

# ╔═╡ 8940791f-fcab-466a-9b5f-a971fa5f6e35
selected_echos = echo_times_vec[echo_times_sorted_index]

# ╔═╡ 7875c9ca-83af-4eca-83ba-222033943967
begin
	local n_selected_echos = length(selected_echos)
	local species_to_echos = get_species_to_echos_mtx(n_species, n_selected_echos, selected_echos, weights, df, phi0; relaxation)
	E = ChemCompOp(species_to_echos, nx, ny, n_species, n_selected_echos, nfft_plan)
end

# ╔═╡ 7caebaa1-8a36-464f-97c1-c06a9f1421b4
function make_reco_params(E, nx, ny, n_species; lambda=1e-4, iterations=100)
	params = Dict{Symbol, Any}()
	params[:reco] = "standard"
	params[:reconSize] = (nx, n_species * ny)
	params[:encodingOps] = [E]
	params[:solver] = CGNR
	params[:reg] = [L2Regularization(lambda)]
	params[:iterations] = iterations
	return params
end

# ╔═╡ 178dce94-49bc-4e62-868e-6b0400335daa
begin
	local params = make_reco_params(E, nx, ny, n_species)
	acq_cs_reco = acq_data_from_echo_selection_v2(acqs, echo_time2meas_n_echo_n, selected_echos)
	img = reconstruction(acq_cs_reco, params);
	c_img = reshape(img.data, nx, ny, n_species)
	nothing
end

# ╔═╡ 8886a40c-a1a3-4006-b3b5-2d428f2967ae
let
	# normalization
	img = abs.(c_img)
	img ./= sum(abs.(c_img); dims=3)

	recons = []
	for (i, label) in enumerate(["Water", "Acetone"])
	    p = heatmap(
			img[div(nx, 4):end-div(nx, 4), :, i],
			c = :grays,
			aspect_ratio=1.0,
			title="\n"*label,
			clim=(0, 1),
			titlefontcolor=:white,
			colorbar_tickfontcolor=:white,
			topmargin=10Plots.px
		)
		push!(recons, p)
	end

	concentrations = Dict("water"=>[], "acetone"=>[])
	for (j, species) in enumerate(["acetone", "water"])
		for i=1:2
			x = mean(collect(1:size(masks[i], 1))[vec(any(masks[i], dims=2))]) - div(nx, 4) + (i-1) / 2 * nx/4
			y = mean(collect(1:size(masks[i], 2))[vec(any(masks[i], dims=1))])
			val = mean(img[erode(masks[i]), j])
			push!(concentrations[species], val)
			annotate!(recons[j], y, x, text("#$i: $(round(val, digits=3))", (val > 0.4) ? :black : :white, :center, 8))
		end
	end

	local data = Dict(
		"acetone_c_values_in_mixture"=>abs.(c_img[:, :, 2][erode(masks[1])]),
		"water_c_values_in_mixture"=>abs.(c_img[:, :, 1][erode(masks[1])]),
		"acetone_c_values_in_water"=>abs.(c_img[:, :, 2][erode(masks[2])]),
		"water_c_values_in_water"=>abs.(c_img[:, :, 1][erode(masks[2])]),
	)
	open(joinpath(RESULTS_FOLDER, "abs_concentrations_of_species_wo_fi.json"), "w") do f
	  JSON.print(f, data, 4)
	end

	local p = plot(
	    recons...,
	    layout=(1, n_species),
	    plot_title="\n1H Ratio for Each Species",
	    size=(1920, 1080/2),
	    dpi=300,
	    background_color=:black,
	    plot_titlefontcolor=:white,
	    plot_titlevspan=0.16,
	    plot_titlefontvalign=:bottom,
	    framestyle=:box,
		showaxis=false,
		grid=false,
		colorbar_tickfontcolor =:white,
		foreground_color=:white
	)
	savefig(p, joinpath(RESULTS_FOLDER, "cs-reco.png"))
	p
end

# ╔═╡ fbda9a19-ad63-4922-b4d9-857cb3830002
md"""
# Sensitivity to parametrization
"""

# ╔═╡ 2dc7d185-b019-4ba6-8e80-25bbdf3fec37
let
	for offset_hz=0:-5:-50
		results_subfolder = joinpath(RESULTS_FOLDER, "parametrization-sensitivity", "offset$offset_hz")
		mkpath(results_subfolder)
		# parametrization
		shifted_acetone_freq = acetone_freq + offset_hz*2pi/1000
		df = convert(Vector{Vector{Float32}}, [
		    [water_freq],
		    [shifted_acetone_freq]
		])
		phi0 = convert(Vector{Float32}, [
		    0.0,
		    0.0,
		])
		weights = convert(Vector{Vector{Float32}}, [
			[Parameters.M0_water],
		    [Parameters.M0_acetone],
		])

		f = open(joinpath(results_subfolder, "model_frequencies.json"), "w")
		JSON.print(f, Dict("acetone" => shifted_acetone_freq, "water" => water_freq), 4)
		close(f)

		# operator
		n_selected_echos = length(selected_echos)
		species_to_echos = get_species_to_echos_mtx(n_species, n_selected_echos, selected_echos, weights, df, phi0)
		E = ChemCompOp(species_to_echos, nx, ny, n_species, n_selected_echos, nfft_plan)
		params = make_reco_params(E, nx, ny, n_species;)
		acq_cs_reco = acq_data_from_echo_selection_v2(acqs, echo_time2meas_n_echo_n, selected_echos)

		img = reconstruction(acq_cs_reco, params);
		c_img = reshape(img.data, nx, ny, n_species)

		data = Dict(
			"acetone_c_values_in_mixture"=>abs.(c_img[:, :, 2][erode(masks[1])]),
			"water_c_values_in_mixture"=>abs.(c_img[:, :, 1][erode(masks[1])]),
			"acetone_c_values_in_water"=>abs.(c_img[:, :, 2][erode(masks[2])]),
			"water_c_values_in_water"=>abs.(c_img[:, :, 1][erode(masks[2])]),
		)
		open(joinpath(results_subfolder, "abs_concentrations_of_species_wo_fi.json"), "w") do f
		  JSON.print(f, data, 4)
		end
	end
end

# ╔═╡ Cell order:
# ╠═659762fb-ba8e-4ae4-b1f7-6c209ac717b2
# ╠═bdaba444-746d-4c14-a707-432fb3d9a03f
# ╠═355a9914-dbdb-49d0-9186-711c7ac1e500
# ╠═bd9c7fc3-7a86-4d8e-8e39-bc6763168cdd
# ╟─5bd490cd-316f-40cc-8354-e25b8659c6af
# ╠═23a171d3-6c7b-4fe6-9a3c-824c969fdd54
# ╠═775edc9a-aae5-4e6f-b465-f54f07e20923
# ╠═444ac30f-bf7a-47b2-a59c-a1c6eec81cdc
# ╟─48499038-a513-4871-96a9-f359f6732122
# ╠═a18153a4-f78c-4ac7-b605-53602cf5f96e
# ╠═e6d8d705-5558-4dfd-a999-f140984e0398
# ╠═d5749a21-3b8b-4141-aee8-dd25cd3f2c40
# ╠═c507bfdd-7e9c-4d53-8b0c-6c31b28f66ce
# ╟─87e61d59-a098-476d-bdec-b88864f1a9f3
# ╠═714619d9-e6d7-4d91-ad91-e616da9b2bb1
# ╠═e09ee2cf-ec4c-4c67-89b9-e263751a3a14
# ╠═4d85e386-ce43-4f62-bdfa-81c53218ec66
# ╟─9b8342d4-8014-4129-8e4f-f10d31249858
# ╟─5ead8775-8a12-4a4e-8e17-f6cf110d0892
# ╠═0e7f25ae-887d-459e-a8f2-f1ecd1905768
# ╟─3be0cb02-2192-4866-8173-a5fff6f5a0bc
# ╠═6ea2bffc-6cf4-4f56-aec9-02d0048336e3
# ╠═8ac278be-d2b7-4ab5-bde2-38fb9c231eb0
# ╠═4766f241-ba3d-4ef8-85dc-84116e0ee161
# ╟─5051eb11-3251-495e-ace0-3e3679220fea
# ╠═cada972b-655a-4519-8e43-d588d5b63a86
# ╟─a577da49-df44-4468-97b6-6a4fd1207a0f
# ╠═04d5f030-0115-4cd2-bcc4-57659d4fcd23
# ╟─6d6d0e0d-7aad-49c7-ad45-298928745d1a
# ╠═8940791f-fcab-466a-9b5f-a971fa5f6e35
# ╠═7875c9ca-83af-4eca-83ba-222033943967
# ╠═7caebaa1-8a36-464f-97c1-c06a9f1421b4
# ╠═178dce94-49bc-4e62-868e-6b0400335daa
# ╠═8886a40c-a1a3-4006-b3b5-2d428f2967ae
# ╟─fbda9a19-ad63-4922-b4d9-857cb3830002
# ╠═2dc7d185-b019-4ba6-8e80-25bbdf3fec37
