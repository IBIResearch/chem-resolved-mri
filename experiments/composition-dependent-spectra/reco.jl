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
	EXPERIMENT_NAME = "composition-dependent-spectra"
	RAW_DATA_FOLDER = joinpath(DATA_FOLDER, "raw")
	RESULTS_FOLDER = joinpath(DATA_FOLDER, "results", EXPERIMENT_NAME)
	mkpath(RESULTS_FOLDER)

	n_species = 2
end

# ╔═╡ 5bd490cd-316f-40cc-8354-e25b8659c6af
md"""
# Data Loading
"""

# ╔═╡ 775edc9a-aae5-4e6f-b465-f54f07e20923
begin
	acqs = []
	echo_times = []
	echo_time2meas_n_echo_n = Dict{Float64, Tuple{Int, Int}}()
	for (i, meas_file_name) in enumerate(["composition-dependent-spectra.h5"])
		mge_meas_path = joinpath(RAW_DATA_FOLDER, meas_file_name)

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
	npzwrite(joinpath(RESULTS_FOLDER, "echo-recons-magnitude.npy"), abs.(direct_reco)[div(nx, 4):end-div(nx, 4), :, :])
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
	(CartesianIndex(60, 46), 1),
	(CartesianIndex(80, 51), 2),
	(CartesianIndex(86, 31), 3),
	(CartesianIndex(65, 25), 4),
	(CartesianIndex(73, 38), 5),
	(CartesianIndex(1, 1), 6)
])

# ╔═╡ e09ee2cf-ec4c-4c67-89b9-e263751a3a14
begin
	magnitude_image = mean(abs.(direct_reco); dims=3)[:, :, 1]
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
	    [-1.735], # acetone
	])
	phi0 = convert(Vector{Float32}, [
	    0.0,
	    0.0,
	])
	weights = convert(Vector{Vector{Float32}}, [
	    [Parameters.M0_water],
	    [Parameters.M0_acetone],
	])
end

# ╔═╡ 6d6d0e0d-7aad-49c7-ad45-298928745d1a
md"""
# Chemically Resolved Reconstruction w/o FI Correction
"""

# ╔═╡ 8940791f-fcab-466a-9b5f-a971fa5f6e35
selected_echos = echo_times_vec[echo_times_sorted_index]

# ╔═╡ 7875c9ca-83af-4eca-83ba-222033943967
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
	npzwrite(joinpath(RESULTS_FOLDER, "c_img.npy"), c_img)
	nothing
end

# ╔═╡ 8886a40c-a1a3-4006-b3b5-2d428f2967ae
begin
	recons = [
	    heatmap(
		        ((abs.(c_img) ./ sum(abs.(c_img); dims=3)) .* sum(masks[1:5]))[div(nx, 4):end-div(nx, 4),:,i] ,
		        c = :grays,
		        aspect_ratio=1.0,
		        title="\n"*label,
				clim=(0, 1),
		        titlefontcolor=:white,
				colorbar_tickfontcolor =:white,
				topmargin=10Plots.px
		    )
		for (i, label) in enumerate(["Water", "Acetone"])
	]

	concentrations = Dict("water"=>[], "acetone"=>[])
	for (j, species) in enumerate(["water", "acetone"])
		for i=1:5
			x = mean(collect(1:size(masks[i], 1))[vec(any(masks[i], dims=2))]) - div(nx, 4)
			y = mean(collect(1:size(masks[i], 2))[vec(any(masks[i], dims=1))])
			val = mean((abs.(c_img) ./ sum(abs.(c_img); dims=3))[erode(masks[i]), j])
			push!(concentrations[species], val)
			annotate!(recons[j], y, x, text("#$i: $(round(val, digits=3))", (val > 0.4) ? :black : :white, :center, 8))
		end
	end
	open(joinpath(RESULTS_FOLDER, "mean-concentrations-cs-reco.json"), "w") do f
	  JSON.print(f, concentrations, 4)
	end

	local p = plot(
	    recons...,
	    layout=(1, 2),
	    plot_title="\n1H Ratio for Each Species",
	    size=(1920/2, 1080/2),
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

# ╔═╡ ff762675-4cbd-41cd-86e4-50101de4112e
f_bias = mean(water_slope_maps[masks[1] .| masks[2] .| masks[3] .| masks[4], :, 1])

# ╔═╡ 45af5d1e-6876-48bd-9cf3-a895fbb492cf
-1.735 - f_bias

# ╔═╡ 2ee60c70-0664-4b05-8ca7-d15e5535430c
begin
	df_fi = convert(Vector{Vector{Float32}}, [
	    [0.0],
	    [-1.735 - f_bias],
	])

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

# ╔═╡ e14a9b4a-4ac8-45f7-a35a-89141b54af29
begin
	local acetone = abs.(c_img_fi)[:, :, 2] ./ 6
	local water = abs.(c_img_fi)[:, :, 1] ./ 2
	acetone_ratio = acetone ./ (acetone + water) .* sum(masks[1:5]);
end

# ╔═╡ 11ed06aa-09ba-4246-92eb-2f4bc1e1e8a4
begin
	local recons = [
	    heatmap(
		        ((abs.(c_img_fi) ./ sum(abs.(c_img_fi); dims=3)) .* sum(masks[1:5]))[div(nx, 4):end-div(nx, 4),:,i] ,
		        c = :grays,
		        aspect_ratio=1.0,
		        title="\n"*label,
				clim=(0, 1),
		        titlefontcolor=:white,
				colorbar_tickfontcolor =:white,
				topmargin=10Plots.px
		    )
		for (i, label) in enumerate(["Water", "Acetone"])
	]

	concentrations_fi = Dict("water"=>[], "acetone"=>[])
	for (j, species) in enumerate(["water", "acetone"])
		for i=1:5
			x = mean(collect(1:size(masks[i], 1))[vec(any(masks[i], dims=2))]) - div(nx, 4)
			y = mean(collect(1:size(masks[i], 2))[vec(any(masks[i], dims=1))])
			val = mean((abs.(c_img_fi) ./ sum(abs.(c_img_fi); dims=3))[erode(masks[i]), j])
			push!(concentrations_fi[species], val)
			annotate!(recons[j], y, x, text("#$i: $(round(val, digits=3))", (val > 0.4) ? :black : :white, :center, 8))
		end
	end
	open(joinpath(RESULTS_FOLDER, "mean-concentrations-cs-reco-fi.json"),"w") do f
	  JSON.print(f, concentrations_fi, 4)
	end

	local p = plot(
	    recons...,
	    layout=(1, 2),
	    plot_title="\n1H Ratio for Each Species",
	    size=(1920/2, 1080/2),
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

# ╔═╡ ad183926-1d3d-4c0e-9d69-93e9c02e0f7d
begin
	n_h2o = [4.442, 3.331, 2.221, 1.110]
	n_ac = [0.272, 0.544, 0.816, 1.088]

	gt_ratios = n_ac ./ (n_ac .+ n_h2o)
	gt_ratios = gt_ratios
end

# ╔═╡ 59384357-c25a-41ff-8d9b-219a17edaf45
gt_ratios

# ╔═╡ 54ca28cc-98cf-477d-9293-1b4d3baa7be6
begin
	acetone_ratios_fi = []
	for i=1:4
		ratio_values = acetone_ratio[erode(masks[i])]
		push!(acetone_ratios_fi, ratio_values)
	end
end

# ╔═╡ 53d52f57-2394-4e88-aec6-fb5c77aa636c
begin
	xtick = collect(0.0:0.1:1.0)
	x = gt_ratios.*10
	y = hcat([r[1:95] for r in acetone_ratios_fi]...)
	p_acetone = boxplot(repeat(x, inner=95), vec(y), legend=false, xticks = (xtick*10, xtick))
	plot!(p_acetone, [0, 0.6].*10, [0, 0.6], color="red")
	p_acetone = plot!(p_acetone, xlabel="GT ratios", ylabel="Estimated ratios", title="Acetone")
end

# ╔═╡ 53daaea6-3f6e-40aa-85b0-4e930189894c
md"""
# Linear approx of the solution of ratio-dependent spectra
"""

# ╔═╡ a17a9f0f-bd57-4a0e-a6d2-755a0d6a76b3
function run_reco(df)
	cs_mtx = zeros(ComplexF32, n_species, length(selected_echos), length(selected_echos))
	for (i, echo_time) in enumerate(selected_echos)
	    for k=1:length(df)
	        for j=1:length(df[k])
	            cs_mtx[k, i, i] += weights[k][j]*exp.(df[k][j]*echo_time*im)
	        end
	        cs_mtx[k, i, i] *= exp(phi0[k]*im)
	    end
	end
	cs_mtx = reshape(cs_mtx, n_species*length(selected_echos), length(selected_echos));
	local E_fi = FieldInhomChemCompOp(nx, ny, n_species, length(selected_echos), nfft_operators, cs_mtx)

	local params = Dict{Symbol, Any}()
	params[:reco] = "standard"
	params[:reconSize] = (nx, n_species*ny)
	params[:encodingOps] = [E_fi]
	params[:solver] = CGNR
	params[:reg] = [L2Regularization(1e-4)]
	params[:iterations] = 100

	img = reconstruction(acq_cs_reco, params);
	c_img = reshape(img.data, nx, ny, n_species)
end

# ╔═╡ 705ba3fc-7c55-477a-998f-7249f1f71fa4
begin
	local acq = acq_data_from_echo_selection_v2(acqs, echo_time2meas_n_echo_n, selected_echos)
	acetone_shifts_range = LinRange(-2.0, -1.0, 10)
	multiple_independent_recos = []
	for _df_acetone in acetone_shifts_range
		local df = convert(Vector{Vector{Float32}}, [
		    [0.0],
		    [_df_acetone]
		])
		local c = run_reco(df)
		push!(multiple_independent_recos, c)
	end
end

# ╔═╡ ba4f0f2a-729a-416c-b823-a6b1481b2e75
begin
	multiple_independent_recos_cat = cat(multiple_independent_recos..., dims=4)
	max_idx = argmax(abs.(multiple_independent_recos_cat)[:, :, 2, :]; dims=3)
	c_img_linear_approx_ratio_dependent = cat([multiple_independent_recos_cat[:, :, k, :][max_idx] for k=1:n_species]..., dims=3)
	npzwrite(RESULTS_FOLDER*"/c_img_linear_approx_ratio_dependent.npy", c_img_linear_approx_ratio_dependent)
end

# ╔═╡ 03f8ea17-ba8e-4ab7-b72b-5d760371d374
let
	p = [
		heatmap(
			abs.(multiple_independent_recos[i][:, :, 2])[div(nx, 4):end-div(nx, 4),:],
			c = :grays,
			aspect_ratio=1.0,
			clim=(0, 1),
			colorbar=false
		)
		for i in 1:length(multiple_independent_recos)
	]
	plot(
		p...,
		layout=(2, div(length(multiple_independent_recos), 2)),
		plot_title="\nAcetone recons for different freq shifts",
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
end

# ╔═╡ 40513b75-407b-4885-869d-0eb79a9d6ba3
begin
	local acetone = abs.(c_img_linear_approx_ratio_dependent)[:, :, 2] ./ 6
	local water = abs.(c_img_linear_approx_ratio_dependent)[:, :, 1] ./ 2
	acetone_ratio_dynamic_lin_approx = acetone ./ (acetone + water) .* sum(masks[1:5]);
end

# ╔═╡ f7a958fd-f66e-409e-89c4-1c075c7a0144
begin
	local recons = [
	    heatmap(
		        ((abs.(c_img_linear_approx_ratio_dependent) ./ sum(abs.(c_img_linear_approx_ratio_dependent); dims=3)) .* sum(masks[1:5]))[div(nx, 4):end-div(nx, 4),:,i] ,
		        c = :grays,
		        aspect_ratio=1.0,
		        title="\n"*label,
				clim=(0, 1),
		        titlefontcolor=:white,
				colorbar_tickfontcolor =:white,
				topmargin=10Plots.px
		    )
		for (i, label) in enumerate(["Water", "Acetone"])
	]

	local concentrations_fi_dynamic = Dict("water"=>[], "acetone"=>[])
	for (j, species) in enumerate(["water", "acetone"])
		for i=1:5
			x = mean(collect(1:size(masks[i], 1))[vec(any(masks[i], dims=2))]) - div(nx, 4)
			y = mean(collect(1:size(masks[i], 2))[vec(any(masks[i], dims=1))])
			val = mean((abs.(c_img_linear_approx_ratio_dependent) ./ sum(abs.(c_img_linear_approx_ratio_dependent); dims=3))[erode(masks[i]), j])
			push!(concentrations_fi_dynamic[species], val)
			annotate!(recons[j], y, x, text("#$i: $(round(val, digits=3))", (val > 0.4) ? :black : :white, :center, 8))
		end
	end
	open(joinpath(RESULTS_FOLDER, "mean-concentrations-cs-reco-fi-dynamic.json"),"w") do f
	  JSON.print(f, concentrations_fi_dynamic, 4)
	end

	local p = plot(
	    recons...,
	    layout=(1, 2),
	    plot_title="\n1H Ratio for Each Species",
	    size=(1920/2, 1080/2),
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
	savefig(p, joinpath(RESULTS_FOLDER, "cs-reco-fi-linear-approx-ratio-dependent.png"))
	p
end

# ╔═╡ 3075012e-4885-4ef2-bd62-cbf20ebc03bc
begin
	acetone_ratios_fi_dynamic_lin_approx = []
	for i=1:4
		ratio_values = acetone_ratio_dynamic_lin_approx[erode(masks[i])]
		push!(acetone_ratios_fi_dynamic_lin_approx, ratio_values)
	end
end

# ╔═╡ fbac7505-51e9-47f9-bb9f-adea893b59d6
let
	xtick = collect(0.0:0.1:1.0)
	x = gt_ratios.*10
	y = hcat([r[1:95] for r in acetone_ratios_fi_dynamic_lin_approx]...)
	p_acetone = boxplot(repeat(x, inner=95), vec(y), legend=false, xticks = (xtick*10, xtick))
	plot!(p_acetone, [0, 0.6].*10, [0, 0.6], color="red")
	p_acetone = plot!(p_acetone, xlabel="GT ratios", ylabel="Estimated ratios", title="Acetone")
end

# ╔═╡ Cell order:
# ╠═659762fb-ba8e-4ae4-b1f7-6c209ac717b2
# ╠═bdaba444-746d-4c14-a707-432fb3d9a03f
# ╠═355a9914-dbdb-49d0-9186-711c7ac1e500
# ╠═bd9c7fc3-7a86-4d8e-8e39-bc6763168cdd
# ╟─5bd490cd-316f-40cc-8354-e25b8659c6af
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
# ╠═cada972b-655a-4519-8e43-d588d5b63a86
# ╟─6d6d0e0d-7aad-49c7-ad45-298928745d1a
# ╠═8940791f-fcab-466a-9b5f-a971fa5f6e35
# ╠═7875c9ca-83af-4eca-83ba-222033943967
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
# ╠═ff762675-4cbd-41cd-86e4-50101de4112e
# ╠═45af5d1e-6876-48bd-9cf3-a895fbb492cf
# ╠═2ee60c70-0664-4b05-8ca7-d15e5535430c
# ╠═35466a05-fa3f-4ff2-91ab-cfd9b94ae3cb
# ╠═0186ef0e-98cd-43e2-afa3-6fc0ae267156
# ╠═1ac2dc4f-3a6b-4020-8603-bbec998f1f45
# ╠═e14a9b4a-4ac8-45f7-a35a-89141b54af29
# ╠═11ed06aa-09ba-4246-92eb-2f4bc1e1e8a4
# ╠═ad183926-1d3d-4c0e-9d69-93e9c02e0f7d
# ╠═59384357-c25a-41ff-8d9b-219a17edaf45
# ╠═54ca28cc-98cf-477d-9293-1b4d3baa7be6
# ╠═53d52f57-2394-4e88-aec6-fb5c77aa636c
# ╟─53daaea6-3f6e-40aa-85b0-4e930189894c
# ╠═a17a9f0f-bd57-4a0e-a6d2-755a0d6a76b3
# ╠═705ba3fc-7c55-477a-998f-7249f1f71fa4
# ╠═ba4f0f2a-729a-416c-b823-a6b1481b2e75
# ╠═03f8ea17-ba8e-4ab7-b72b-5d760371d374
# ╠═40513b75-407b-4885-869d-0eb79a9d6ba3
# ╠═f7a958fd-f66e-409e-89c4-1c075c7a0144
# ╠═3075012e-4885-4ef2-bd62-cbf20ebc03bc
# ╠═fbac7505-51e9-47f9-bb9f-adea893b59d6
