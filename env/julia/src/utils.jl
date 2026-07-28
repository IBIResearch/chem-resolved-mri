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
	total_samples = round(Int, nx / acceleration)
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
