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
	using JLD2
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
	EXPERIMENT_NAME = "B0-map-validation"
	RAW_DATA_FOLDER = joinpath(DATA_FOLDER, "raw")
	RESULTS_FOLDER = joinpath(DATA_FOLDER, "results", EXPERIMENT_NAME)
	mkpath(RESULTS_FOLDER)

	n_species = 2
end

# ╔═╡ b0486a0e-ac8e-4eb3-ad5e-667d10eac02e
function filter_noisy_dummy_measurements(raw::RawAcquisitionData)
    ACQ_IS_NOISE_MEASUREMENT = 2<<17
    ACQ_IS_DUMMYSCAN_DATA = 2<<25
	ISMRMRD_ACQ_IS_NAVIGATION_DATA = 2 << 22
    return RawAcquisitionData(
        raw.params,
        filter(x -> x.head.flags & (ACQ_IS_NOISE_MEASUREMENT | ACQ_IS_DUMMYSCAN_DATA | ISMRMRD_ACQ_IS_NAVIGATION_DATA) == 0, raw.profiles),
    )
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

	for (i, meas_file_name) in enumerate(["B0-map-validation.h5"])
		mge_meas_path = joinpath(RAW_DATA_FOLDER, meas_file_name)

		raw = RawAcquisitionData(ISMRMRDFile(mge_meas_path))
		raw = filter_noisy_dummy_measurements(raw)
		acq = AcquisitionData(raw)
	    push!(acqs, acq)


		push!(echo_times, raw.params["TE"])


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
	(CartesianIndex(64, 35), 1),
	(CartesianIndex(80, 17), 2),
	(CartesianIndex(94, 41), 3),
	(CartesianIndex(77, 54), 4),
	(CartesianIndex(80, 40), 5),
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
	    if label == 5
	        mask = mask .& (.~dilate(masks[1] .| masks[2] .| masks[3] .| masks[4]))
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

# ╔═╡ 7d2973db-3229-451a-ba40-c1203d1f3024
md"""
# Estimation of Field Inhomogeniety Map
"""

# ╔═╡ 15150fb7-8279-41a9-af77-c8c2028ae94b
function estimate_slopes(mask; recos=recos)
	cat_recos = permutedims(cat(recos..., dims=4), [1, 2, 4, 3]);
	water_phases_for_field_inhomogeniety = angle.(cat_recos)[mask, :, :]
	for i=1:size(water_phases_for_field_inhomogeniety, 1)
	    for j=1:size(water_phases_for_field_inhomogeniety, 2)
	        water_phases_for_field_inhomogeniety[i, j, 1:2:32] .= unwrap(water_phases_for_field_inhomogeniety[i, j, 1:2:32])
			water_phases_for_field_inhomogeniety[i, j, 2:2:32] .= unwrap(water_phases_for_field_inhomogeniety[i, j, 2:2:32])
	    end
	end

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

	return water_phases_slopes
end

# ╔═╡ 27dc2b8e-d92e-4f05-ade9-0a4217c766bc
water_phases_slopes = estimate_slopes(masks[5]; recos=recos)

# ╔═╡ 946ca5df-a4a1-408d-8030-71250718b4d4
begin
	local plots = []
	local water_mask = masks[5]
	for i=1:size(water_phases_slopes, 2)
		for j=1:2
		    slopes_map = zeros(Float64, nx, ny)
		    slopes_map[water_mask] .= water_phases_slopes[:, i, j]
			suffix = (j == 1) ? "odd" : "even"
			npzwrite(joinpath(RESULTS_FOLDER, "slopes_map_$suffix.npy"), slopes_map)
		    p = heatmap(slopes_map[div(nx, 4):end-div(nx, 4), :], aspect_ratio=1, color=:grays, showaxis=false, colorbar=false, title="Measurement #$i: $suffix")
		    push!(plots, p)
		end
	end

	slopes_diff = water_phases_slopes[:, 1, 1] .- water_phases_slopes[:, 1, 2]
	slopes_diff_map = zeros(Float64, nx, ny)
	slopes_diff_map[water_mask] .= slopes_diff
	push!(plots, heatmap(slopes_diff_map[div(nx, 4):end-div(nx, 4), :], aspect_ratio=1, color=:grays, showaxis=false, colorbar=false, title="Slope Diff Odd/Even"))


	local p = plot(
	    plots...,
	    layout=(1, 3),
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

# ╔═╡ 32798032-a3de-44f6-acfe-bd6262d4c486
npzwrite(joinpath(RESULTS_FOLDER, "water_phases_slopes_water_cyliner.npy"), water_phases_slopes)

# ╔═╡ 2757a590-600e-496d-9d87-c77658fb5597
md"""
## Inpainiting for tube inserts
"""

# ╔═╡ 8ef41b7f-4916-479d-a0df-a0572986460c
begin
	local water_mask = masks[5]
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
		end
	end
end

# ╔═╡ d1bac90d-b6d8-4d5a-81fb-af26bbc81b41
heatmap(
	water_slope_maps[div(nx, 4):end-div(nx, 4), :, 1, 1],
	aspect_ratio=1, color=:grays, showaxis=false, colorbar=false,
	title="Field Inhogenity map", grid=false
)

# ╔═╡ c45605e3-3208-4328-9f6f-1d591fb2aefe
npzwrite(joinpath(RESULTS_FOLDER, "water_phases_slopes_water_cyliner_extrapolated.npy"), water_slope_maps)

# ╔═╡ 88dfdb34-04ae-4cc5-9219-cbcdba99a91c
for i=1:4
	slopes_tube = estimate_slopes(masks[i])
	npzwrite(joinpath(RESULTS_FOLDER, "water_phases_slopes_tube_$i.npy"), slopes_tube)
end

# ╔═╡ Cell order:
# ╠═659762fb-ba8e-4ae4-b1f7-6c209ac717b2
# ╠═bdaba444-746d-4c14-a707-432fb3d9a03f
# ╠═355a9914-dbdb-49d0-9186-711c7ac1e500
# ╠═bd9c7fc3-7a86-4d8e-8e39-bc6763168cdd
# ╠═b0486a0e-ac8e-4eb3-ad5e-667d10eac02e
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
# ╟─7d2973db-3229-451a-ba40-c1203d1f3024
# ╠═15150fb7-8279-41a9-af77-c8c2028ae94b
# ╠═27dc2b8e-d92e-4f05-ade9-0a4217c766bc
# ╠═946ca5df-a4a1-408d-8030-71250718b4d4
# ╠═32798032-a3de-44f6-acfe-bd6262d4c486
# ╟─2757a590-600e-496d-9d87-c77658fb5597
# ╠═8ef41b7f-4916-479d-a0df-a0572986460c
# ╠═d1bac90d-b6d8-4d5a-81fb-af26bbc81b41
# ╠═c45605e3-3208-4328-9f6f-1d591fb2aefe
# ╠═88dfdb34-04ae-4cc5-9219-cbcdba99a91c
