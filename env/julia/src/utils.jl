using Colors
using Random
using StableRNGs


function mask_to_rgb(mask, color)
    overlay = fill(RGBA(0, 0, 0, 0), size(mask))
    for i in 1:size(mask, 1)
        for j in 1:size(mask, 2)
            if mask[i, j]
                overlay[i, j] = color
            end
        end
    end
    return overlay
  end


function undersampling_mask(nx, ny, acceleration, center_fraction; rng=StableRNG(42))
	mask = falses(ny)
	total_samples = round(Int, ny / acceleration)
	num_center = round(Int, ny * center_fraction)
	center_start = div(ny - num_center, 2) + 1
	center_end = center_start + num_center - 1

	mask[center_start:center_end] .= true

	candidates = vcat(1:center_start-1, center_end+1:ny)
	remaining = total_samples - num_center
	@assert remaining > 0
	selected = randperm(rng, length(candidates))[1:remaining]
	mask[selected] .= true
	mask = repeat(reshape(mask, 1, ny), outer=nx)
end


function get_arg(default::String)
	# When running as a script: arguments are in ARGS
	if @isdefined PlutoRunner
		# Inside Pluto → use default
		return default
	else
		# As a script → use ARGS[1] if available
		return get(ARGS, 1, default)
	end
end


function filter_outliers(x; k=1.5)
    q1, q3 = quantile(x, [0.25, 0.75])
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    return x[(x .>= lower) .& (x .<= upper)]
end


function filter_noisy_dummy_measurements(raw::RawAcquisitionData)
    ACQ_IS_NOISE_MEASUREMENT = 2<<17
    ACQ_IS_DUMMYSCAN_DATA = 2<<25
	ISMRMRD_ACQ_IS_NAVIGATION_DATA = 2 << 22
    return RawAcquisitionData(
        raw.params,
        filter(x -> x.head.flags & (ACQ_IS_NOISE_MEASUREMENT | ACQ_IS_DUMMYSCAN_DATA | ISMRMRD_ACQ_IS_NAVIGATION_DATA) == 0, raw.profiles),
    )
end
