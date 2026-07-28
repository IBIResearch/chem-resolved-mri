using MRIBase


function invert_readout_for_odd_echos!(acqs; nc=2)
	nx, ny = acqs[1].encodingSize[1], acqs[1].encodingSize[2]

    for acq in acqs
        for i in CartesianIndices(acq.kdata)
            if i[1] % 2 == 1
                acq.kdata[i] .= reshape(reshape(acq.kdata[i], nx, ny, nc)[end:-1:begin, :, :], nx*ny, nc)
            end
        end
    end
end


function shift_ky!(acqs; nc=2)
	nx, ny = acqs[1].encodingSize[1], acqs[1].encodingSize[2]

    nodes = MRIBase.cartesian2dNodes(
		Float32, ny, nx; 
		kmin=(-0.5, -0.5), kmax=(0.5, 0.5)
	)
	_ky = reshape(nodes, 2, nx, ny)[2, :, :]
	
	for acq in acqs
	    for i in CartesianIndices(acq.kdata)
	        k = reshape(acq.kdata[i], nx, ny, nc)
	        for i=1:nc
	            k[:, :, i] .*= exp.((im * 2pi * ny/2) .* _ky)
	        end
	        acq.kdata[i] .= reshape(k, nx*ny, nc)
	    end
	end
end

function acq_data_from_echo_selection_v2(acqs, echo_time2meas_n_echo_n, selected_echos)
	nx, ny = acqs[1].encodingSize[1], acqs[1].encodingSize[2]
	n_points_per_profile = nx
	n_profiles = ny

	# constructing trajectory
	timings = vcat([echo .+ zeros(nx*ny) for echo in selected_echos]...)
	timings = convert(Vector{Float32}, timings);

	nodes = MRIBase.cartesian2dNodes(Float32, n_profiles, n_points_per_profile; kmin=(-0.5, -0.5), kmax=(0.5, 0.5))
	nodes = reshape(nodes, 2, n_points_per_profile, n_profiles)
	nodes = repeat(nodes, outer=(1, 1, length(selected_echos)))
	nodes = reshape(nodes, 2, n_points_per_profile*n_profiles*length(selected_echos));

	traj = Trajectory(
	    nodes,  
	    n_profiles, 
	    n_points_per_profile; 
	    times=timings, 
	    cartesian=true
	)

	# constructing kdata
	new_kdata = Array{ComplexF32}(undef, n_points_per_profile, n_profiles, length(selected_echos))
	for (i, echo)=enumerate(selected_echos)
	    meas_n, echo_n_within_meas = echo_time2meas_n_echo_n[echo]
	    kspace_per_echo = kDataCart(acqs[meas_n])[:, :, 1, 1, echo_n_within_meas, 1]
	    new_kdata[:, :, i] = kspace_per_echo
	end
	new_kdata = vec(new_kdata)
	new_kdata = reshape(new_kdata, (size(new_kdata)..., 1))

	return AcquisitionData(
		traj, [new_kdata for i=1:1,j=1:1,k=1:1]; encodingSize=acqs[1].encodingSize, fov=acqs[1].fov
	)
end