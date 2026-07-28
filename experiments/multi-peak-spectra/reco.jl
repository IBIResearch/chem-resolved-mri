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
	EXPERIMENT_NAME = "multi-peak-spectra"
	RAW_DATA_FOLDER = joinpath(DATA_FOLDER, "raw")
	RESULTS_FOLDER = joinpath(DATA_FOLDER, "results", EXPERIMENT_NAME)
	mkpath(RESULTS_FOLDER)

	n_species = 3
end

# ╔═╡ 5bd490cd-316f-40cc-8354-e25b8659c6af
md"""
# Data Loading
"""

# ╔═╡ 6a3a1c67-d1b9-4f81-ab05-92c869a9c191
function filter_noisy_dummy_measurements(raw::RawAcquisitionData)
    ACQ_IS_NOISE_MEASUREMENT = 2<<17
    ACQ_IS_DUMMYSCAN_DATA = 2<<25
	ISMRMRD_ACQ_IS_NAVIGATION_DATA = 2 << 22
    return RawAcquisitionData(
        raw.params,
        filter(x -> x.head.flags & (ACQ_IS_NOISE_MEASUREMENT | ACQ_IS_DUMMYSCAN_DATA | ISMRMRD_ACQ_IS_NAVIGATION_DATA) == 0, raw.profiles),
    )
end

# ╔═╡ 775edc9a-aae5-4e6f-b465-f54f07e20923
begin
	acqs = []
	echo_times = []
	f0 = []
	echo_time2meas_n_echo_n = Dict{Float64, Tuple{Int, Int}}()
	for (i, meas_file_name) in enumerate(["multipeak_spectra.h5"])
		mge_meas_path = joinpath(RAW_DATA_FOLDER, meas_file_name)

		raw = RawAcquisitionData(ISMRMRDFile(mge_meas_path))
		raw = filter_noisy_dummy_measurements(raw)
		acq = AcquisitionData(raw)
	    push!(acqs, acq)

		echos = raw.params["TE"]
		push!(f0, raw.params["H1resonanceFrequency_Hz"])

		push!(echo_times, echos)
		for (echo_n, t) in enumerate(echos)
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

	# additional pre-processing
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
	    # we take only the first coil because coils differ in phase
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
	        angle.(direct_reco)[div(nx, 4):end-div(nx, 4), :, i] .* (abs.(direct_reco)[div(nx, 4):end-div(nx, 4), :, 4] .> 0.005),
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
	(CartesianIndex(83, 25), 1),
	(CartesianIndex(64, 41), 2),
	(CartesianIndex(80, 56), 3),
	(CartesianIndex(97, 39), 4),
	(CartesianIndex(81, 39), 5),
	(CartesianIndex(1, 1), 6)
])

# ╔═╡ e09ee2cf-ec4c-4c67-89b9-e263751a3a14
begin
	magnitude_image = abs.(adjoint(nfft_plan) * vec(mean(kDataCart(acqs[1])[:, :, 1, :, 1, 1]; dims=3)))
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

	for label in 1:5
	    mask = labels_map(segments) .== label
	    if label < 5
	        mask = erode(mask)
	    end
	    push!(masks, mask)
	    mask = mask[div(nx, 4):end-div(nx, 4), :]
	    overlay = mask_to_rgb(mask, RGBA(get(ColorSchemes.lajolla100, label/length(seeds)), 0.2))
	    p = plot!(p, overlay)

	    mask_center = mean(hcat([[c[1], c[2]] for c in findall(mask)]...), dims=2)
	    annotate!(p, mask_center[2], mask_center[1], text("#$label", :black, :center, 10))
	end
	npzwrite(joinpath(RESULTS_FOLDER, "masks.npy"), cat(masks...; dims=3))
	p
end

# ╔═╡ 9b8342d4-8014-4129-8e4f-f10d31249858
md"""
# Forward Model
"""

# ╔═╡ cada972b-655a-4519-8e43-d588d5b63a86
begin
	df = convert(Vector{Vector{Float32}}, [
	    [0.0], # water
	    [-198 * 2pi / 1000], # acetone
		[-396.511, -79.3021, 158.604] .* 2pi ./ 1000
	])
	phi0 = convert(Vector{Float32}, [
	    0.0,
	    0.0,
		0.0
	])
	weights = convert(Vector{Vector{Float32}}, [
	    [Parameters.M0_water],
	    [Parameters.M0_acetone],
		Parameters.M0_ethanol .* Parameters.ethanol_normalized_weights,
	])
end

# ╔═╡ ce189931-993b-4199-9fbc-7abdafa99a54
df

# ╔═╡ 6d6d0e0d-7aad-49c7-ad45-298928745d1a
md"""
# Chemically Resolved Reconstruction w/o FI Correction
"""

# ╔═╡ 8940791f-fcab-466a-9b5f-a971fa5f6e35
selected_echos = echo_times_vec[echo_times_sorted_index]

# ╔═╡ e0753984-4e5f-4be5-b26c-e6d58bd2f130
begin
	local n_selected_echos = length(selected_echos)
	local species_to_echos = get_species_to_echos_mtx(n_species, n_selected_echos, selected_echos, weights, df, phi0)
	E = ChemCompOp(species_to_echos, nx, ny, n_species, n_selected_echos, nfft_plan)
end

# ╔═╡ 178dce94-49bc-4e62-868e-6b0400335daa
begin
	local params = Dict{Symbol, Any}()
	params[:reco] = "standard"
	params[:reconSize] = (nx, n_species*ny)
	params[:encodingOps] = [E]
	params[:solver] = CGNR
	params[:reg] = [L2Regularization(1e-4)]
	params[:iterations] = 100

	acq_cs_reco = acq_data_from_echo_selection_v2(acqs, echo_time2meas_n_echo_n, selected_echos)

	img = reconstruction(acq_cs_reco, params);
	c_img = reshape(img.data, nx, ny, n_species)

	nothing
end

# ╔═╡ 8886a40c-a1a3-4006-b3b5-2d428f2967ae
let
	recons = []
	for (i, label) in enumerate(["Water", "Acetone", "Ethanol"])
	    p = heatmap(
			((abs.(c_img) ./ sum(abs.(c_img); dims=3)) .* sum(masks[1:5]))[div(nx, 4):end-div(nx, 4),:,i],
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

	concentrations = Dict("water"=>[], "acetone"=>[], "ethanol"=>[])
	for (j, species) in enumerate(["water", "acetone", "ethanol"])
		for i=1:5
			x = mean(collect(1:size(masks[i], 1))[vec(any(masks[i], dims=2))]) - div(nx, 4)
			y = mean(collect(1:size(masks[i], 2))[vec(any(masks[i], dims=1))])
			img = ((abs.(c_img) ./ sum(abs.(c_img); dims=3)) .* sum(masks[1:5]))[:,:,j]
			val = mean(img[erode(masks[i])])
			push!(concentrations[species], val)
			annotate!(recons[j], y, x, text("#$i: $(round(val, digits=3))", (val > 0.4) ? :black : :white, :center, 8))
		end
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

# ╔═╡ 7d2973db-3229-451a-ba40-c1203d1f3024
md"""
# Estimation of Field Inhomogeniety Map
"""

# ╔═╡ c3f5251d-f48e-4a17-9c89-cd995e07c554
water_mask = masks[5];

# ╔═╡ a04229dc-3f64-4677-8947-ad58d156f11a
begin
	cat_recos = permutedims(cat(recos..., dims=4), [1, 2, 4, 3]);
	water_phases_for_field_inhomogeniety = angle.(cat_recos)[masks[5], :, :]
	for i=1:size(water_phases_for_field_inhomogeniety, 1)
	    for j=1:size(water_phases_for_field_inhomogeniety, 2)
	        water_phases_for_field_inhomogeniety[i, j, 1:2:32] .= unwrap(water_phases_for_field_inhomogeniety[i, j, 1:2:32])
			water_phases_for_field_inhomogeniety[i, j, 2:2:32] .= unwrap(water_phases_for_field_inhomogeniety[i, j, 2:2:32])
	    end
	end
end

# ╔═╡ c17b0037-2e06-4b96-be20-34351b94ee8e
begin
	water_phases_slopes = zeros(Float64, size(water_phases_for_field_inhomogeniety, 1), size(water_phases_for_field_inhomogeniety, 2), 2)
	water_phases_intercepts = zeros(Float64, size(water_phases_for_field_inhomogeniety, 1), size(water_phases_for_field_inhomogeniety, 2), 2)

	for i=1:size(water_phases_for_field_inhomogeniety, 1)
	    for j=1:size(water_phases_for_field_inhomogeniety, 2)
	        echo_times = all_echo_times[j][1:2:32]
	        df = DataFrame(
	            X = echo_times,
	            Y = Float64.(water_phases_for_field_inhomogeniety[i, j, 1:2:32]),
	        )
	        phase_ols = lm(@formula(Y ~ X), df)
	        water_phases_intercepts[i, j, 1] = coef(phase_ols)[1]
	        water_phases_slopes[i, j, 1] = coef(phase_ols)[2]

			echo_times = all_echo_times[j][2:2:32]
	        df = DataFrame(
	            X = echo_times,
	            Y = Float64.(water_phases_for_field_inhomogeniety[i, j, 2:2:32]),
	        )
	        phase_ols = lm(@formula(Y ~ X), df)
	        water_phases_intercepts[i, j, 2] = coef(phase_ols)[1]
	        water_phases_slopes[i, j, 2] = coef(phase_ols)[2]
	    end
	end
end


# ╔═╡ 946ca5df-a4a1-408d-8030-71250718b4d4
begin
	local plots = []
	for i=1:size(water_phases_slopes, 2)
		for j=1:2
		    slopes_map = zeros(Float64, nx, ny)
		    slopes_map[water_mask] .= water_phases_slopes[:, i, j]
			suffix = (j == 1) ? "odd" : "even"
		    p = heatmap(slopes_map[div(nx, 4):end-div(nx, 4), :], aspect_ratio=1, color=:grays, showaxis=false, colorbar=false, title="Measurement #$i: $suffix")
		    push!(plots, p)
		end
	end

	local p = plot(
	    plots...,
	    layout=(1, 2),
	    plot_title="\n Field inhomodeniety map for different measurements",
	    size=(1920, 1080/2),
	    dpi=300,
	    plot_titlevspan=0.08,
	    plot_titlefontvalign=:bottom,
	    framestyle=:box,
	    grid=false
	)
	savefig(p, joinpath(RESULTS_FOLDER, "field-inhomogeniety-map-in-water-per-measurement.png"))
	p
end

# ╔═╡ a9afb4a8-4d66-4e4b-8b84-024874aa9a95
begin
	local plots = []
	for i=1:size(water_phases_intercepts, 2)
		for j=1:2
		    intercepts_map = ones(Float64, nx, ny) .* mean(water_phases_intercepts[:, i, j])
		    intercepts_map[water_mask] .= water_phases_intercepts[:, i, j]
			suffix = (j == 1) ? "odd" : "even"
		    p = heatmap(intercepts_map[div(nx, 4):end-div(nx, 4), :], aspect_ratio=1, color=:grays, showaxis=false, colorbar=false, title="Measurement #$i $suffix")
		    push!(plots, p)
		end
	end

	plot(
	    plots...,
	    layout=(1, 2),
	    plot_title="\n phi0 map for different measurements",
	    size=(1920, 1080/2),
	    dpi=300,
	    plot_titlevspan=0.08,
	    plot_titlefontvalign=:bottom,
	    framestyle=:box,
	    grid=false
	)
end

# ╔═╡ 2757a590-600e-496d-9d87-c77658fb5597
md"""
## Inpainiting for tube inserts
"""

# ╔═╡ 8ef41b7f-4916-479d-a0df-a0572986460c
begin
	n_measurements = length(acqs)

	water_slope_maps = zeros(Float64, nx, ny, n_measurements, 2)
	water_intercepts = zeros(Float64, n_measurements, 2); # intercepts are just constants

	local _grid = MRIBase.cartesian2dNodes(Float32, ny, nx; kmin=(-nx/2, -ny/2), kmax=(nx/2, ny/2)) .* (-2pi)
	_grid = reshape(_grid, 2, nx, ny)
	x1 = _grid[1, :, :][water_mask]
	x2 = _grid[2, :, :][water_mask];

	slope_ols_coefs = zeros(Float64, n_measurements, 2, 6)
	for i=1:n_measurements
		for j=1:2
			# there are outliers in the slopes, as can be seen
			slopes_mask = water_phases_slopes[:, i, j] .> -100

		    df = DataFrame(
				y=water_phases_slopes[:, i, j][slopes_mask],
				x1=x1[slopes_mask],
				x2=x2[slopes_mask],
				x3=x1[slopes_mask].^2,
				x4=x2[slopes_mask].^2,
				x5=x1[slopes_mask].*x2[slopes_mask]
			);
		    slope_ols = lm(term(:y) ~ sum(term.(Symbol.(names(df, Not(:y))))), df)
		    slope_ols_coefs[i, j, :] .= coef(slope_ols)

		    recovered_slopes = predict(slope_ols, DataFrame(
		        x1=vec(_grid[1, :, :]),
		        x2=vec(_grid[2, :, :]),
		        x3=vec(_grid[1, :, :]).^2,
		        x4=vec(_grid[2, :, :]).^2,
		        x5=vec(_grid[1, :, :]).*vec(_grid[2, :, :])
		    ))
			recovered_slopes = reshape(recovered_slopes, nx, ny);

		    water_slope_maps[:, :, i, j] .= recovered_slopes
		    water_intercepts[i, j] = mean(water_phases_intercepts[:, i, j])
		end
	end
end

# ╔═╡ d1bac90d-b6d8-4d5a-81fb-af26bbc81b41
heatmap(
	water_slope_maps[div(nx, 4):end-div(nx, 4), :, 1, 1],
	aspect_ratio=1, color=:grays, showaxis=false, colorbar=false,
	title="Field Inhogenity map", grid=false
)

# ╔═╡ 4169743a-ab2d-44a7-acff-37c54d9da129
md"""
## Chemical Shift Operator with Field Inhomogeniety
"""

# ╔═╡ 00636d1c-0c72-4e2d-9773-7ec85f485879
begin
	local D = 2
	local J = nx*ny
	local N = (nx, ny)

	nfft_operators = []
	phase_offsets = []
	for echo_time in selected_echos
	    measurement_n, echo_n = echo_time2meas_n_echo_n[echo_time]

	    traj = CartesianTrajectory(Float32, ny, nx; TE=echo_time, AQ=1e-4)

	    p = FieldmapNFFTOp(
	        (nx, ny),
	        traj,
	        ComplexF32.(-1im*water_slope_maps[:, :, measurement_n, (echo_n - 1) % 2 + 1]),
	        symmetrize=false,
	        S=Vector{ComplexF32},
	        echoImage=false,
	        alpha=1.75,
	        m=3.0
	    )
	    push!(nfft_operators, p)

	    phase_offset = slope_ols_coefs[measurement_n, (echo_n - 1) % 2 + 1, 1]*echo_time + water_intercepts[measurement_n, (echo_n - 1) % 2 + 1]
	    push!(phase_offsets, phase_offset)
	end
end

# ╔═╡ 3ca22af9-4864-48cd-a71d-ee1453b5e9ae
f_bias = mean(water_slope_maps[masks[1] .| masks[2] .| masks[3] .| masks[4], :, 1])

# ╔═╡ 2ee60c70-0664-4b05-8ca7-d15e5535430c
begin
	df_fi = copy(df)
	df_fi[2] .-= f_bias
	df_fi[3] .-= f_bias

	cs_mtx = zeros(ComplexF32, n_species, length(selected_echos), length(selected_echos))
	for (i, echo_time) in enumerate(selected_echos)
	    for k=1:length(df)
	        for j=1:length(df[k])
	            cs_mtx[k, i, i] += weights[k][j]*exp.(df_fi[k][j]*echo_time*im)
	        end
	        cs_mtx[k, i, i] *= exp(phi0[k]*im)
	    end
	end
	cs_mtx = reshape(cs_mtx, n_species*length(selected_echos), length(selected_echos));
end

# ╔═╡ 35466a05-fa3f-4ff2-91ab-cfd9b94ae3cb
E_fi = FieldInhomChemCompOp(nx, ny, n_species, length(selected_echos), nfft_operators, cs_mtx)

# ╔═╡ 0186ef0e-98cd-43e2-afa3-6fc0ae267156
md"""
# Chemically Resolved Reconstruction w/ FI Correction
"""

# ╔═╡ 1ac2dc4f-3a6b-4020-8603-bbec998f1f45
begin
	local params = Dict{Symbol, Any}()
	params[:reco] = "standard"
	params[:reconSize] = (nx, n_species*ny)
	params[:encodingOps] = [E_fi]
	params[:solver] = CGNR
	params[:reg] = [L2Regularization(1e-4)]
	params[:iterations] = 100

	img_fi = reconstruction(acq_cs_reco, params);
	c_img_fi = reshape(img_fi, nx, ny, n_species);
	npzwrite(joinpath(RESULTS_FOLDER, "c_img_fi.npy"), c_img_fi)
	nothing
end

# ╔═╡ 11ed06aa-09ba-4246-92eb-2f4bc1e1e8a4
let
	recons = []
	for (i, label) in enumerate(["Water", "Acetone", "Ethanol"])
	    p = heatmap(
			((abs.(c_img_fi) ./ sum(abs.(c_img_fi); dims=3)) .* sum(masks[1:5]))[div(nx, 4):end-div(nx, 4),:,i],
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

	concentrations = Dict("water"=>[], "acetone"=>[], "ethanol"=>[])
	for (j, species) in enumerate(["water", "acetone", "ethanol"])
		for i=1:5
			x = mean(collect(1:size(masks[i], 1))[vec(any(masks[i], dims=2))]) - div(nx, 4)
			y = mean(collect(1:size(masks[i], 2))[vec(any(masks[i], dims=1))])
			img = ((abs.(c_img_fi) ./ sum(abs.(c_img_fi); dims=3)) .* sum(masks[1:5]))[:,:,j]
			val = mean(img[erode(masks[i])])
			push!(concentrations[species], val)
			annotate!(recons[j], y, x, text("#$i: $(round(val, digits=3))", (val > 0.4) ? :black : :white, :center, 8))
		end
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
	savefig(p, joinpath(RESULTS_FOLDER, "cs-reco-fi.png"))
	p
end

# ╔═╡ Cell order:
# ╠═659762fb-ba8e-4ae4-b1f7-6c209ac717b2
# ╠═bdaba444-746d-4c14-a707-432fb3d9a03f
# ╠═355a9914-dbdb-49d0-9186-711c7ac1e500
# ╠═bd9c7fc3-7a86-4d8e-8e39-bc6763168cdd
# ╟─5bd490cd-316f-40cc-8354-e25b8659c6af
# ╠═6a3a1c67-d1b9-4f81-ab05-92c869a9c191
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
# ╠═ce189931-993b-4199-9fbc-7abdafa99a54
# ╠═cada972b-655a-4519-8e43-d588d5b63a86
# ╟─6d6d0e0d-7aad-49c7-ad45-298928745d1a
# ╠═8940791f-fcab-466a-9b5f-a971fa5f6e35
# ╠═e0753984-4e5f-4be5-b26c-e6d58bd2f130
# ╠═178dce94-49bc-4e62-868e-6b0400335daa
# ╠═8886a40c-a1a3-4006-b3b5-2d428f2967ae
# ╟─7d2973db-3229-451a-ba40-c1203d1f3024
# ╠═c3f5251d-f48e-4a17-9c89-cd995e07c554
# ╠═a04229dc-3f64-4677-8947-ad58d156f11a
# ╠═c17b0037-2e06-4b96-be20-34351b94ee8e
# ╠═946ca5df-a4a1-408d-8030-71250718b4d4
# ╠═a9afb4a8-4d66-4e4b-8b84-024874aa9a95
# ╟─2757a590-600e-496d-9d87-c77658fb5597
# ╠═8ef41b7f-4916-479d-a0df-a0572986460c
# ╠═d1bac90d-b6d8-4d5a-81fb-af26bbc81b41
# ╟─4169743a-ab2d-44a7-acff-37c54d9da129
# ╠═00636d1c-0c72-4e2d-9773-7ec85f485879
# ╠═3ca22af9-4864-48cd-a71d-ee1453b5e9ae
# ╠═2ee60c70-0664-4b05-8ca7-d15e5535430c
# ╠═35466a05-fa3f-4ff2-91ab-cfd9b94ae3cb
# ╟─0186ef0e-98cd-43e2-afa3-6fc0ae267156
# ╠═1ac2dc4f-3a6b-4020-8603-bbec998f1f45
# ╠═11ed06aa-09ba-4246-92eb-2f4bc1e1e8a4
