using Statistics

function find_local_maxima(y; delta_thres=5)
	maxima_indices = findall(i -> y[i] > y[i-1] && y[i] > y[i+1], 2:length(y)-1) .+ 1
	filtered_maxima_indices = [maxima_indices[1]]
	for i=2:length(maxima_indices)
		if maxima_indices[i] - filtered_maxima_indices[end] < delta_thres
			continue
		end
		push!(filtered_maxima_indices, maxima_indices[i])
	end
	vals = y[filtered_maxima_indices]
	filtered_maxima_indices = filtered_maxima_indices[abs.(vals .- median(vals)) .< std(vals)]
	return filtered_maxima_indices
end


function find_local_minima(y; delta_thres=5)
	maxima_indices = findall(i -> y[i] < y[i-1] && y[i] < y[i+1], 2:length(y)-1) .+ 1
	filtered_maxima_indices = [maxima_indices[1]]
	for i=2:length(maxima_indices)
		if maxima_indices[i] - filtered_maxima_indices[end] < delta_thres
			continue
		end
		push!(filtered_maxima_indices, maxima_indices[i])
	end
	vals = y[filtered_maxima_indices]
	filtered_maxima_indices = filtered_maxima_indices[abs.(vals .- median(vals)) .< std(vals)]
	return filtered_maxima_indices
end


function unwrap(phase_values)
    unwrapped = copy(phase_values)
    for i=2:length(phase_values)
        difference = phase_values[i] - phase_values[i-1]
        if difference > π
            unwrapped[i:end] .-= 2π
        elseif difference < -π
            unwrapped[i:end] .+= 2π
        end
    end
    return unwrapped
end