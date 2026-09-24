using LinearOperators


function forward_model(x, species_to_echos, nx, ny, n_species, n_echos, nfft_plan)
    x = reshape(x, nx, ny, n_species)
    y = zeros(eltype(x), nx, ny, n_species)
    for i=1:n_species
        y[:, :, i] .= reshape(nfft_plan * x[:, :, i], nx, ny)
    end
    y = reshape(y, nx*ny, n_species)
    y = y * species_to_echos
    return vec(y)
end


function conj_forward_model(y, species_to_echos, nx, ny, n_species, n_echos, nfft_plan)
    x = reshape(y, nx*ny, n_echos)
    x = x * species_to_echos'
    x = reshape(x, nx, ny, n_species)
    y = zeros(eltype(x), nx, ny, n_species)
    for i=1:n_species
        y[:, :, i] .= adjoint(nfft_plan) * vec(x[:, :, i])
    end
    # y ./= prod((nx, ny))
    return vec(y)
end


function ChemCompOp(species_to_echos, nx, ny, n_species, n_echos, nfft_plan)
    nrow = nx*ny*n_echos
    ncol = nx*ny*n_species
    return LinearOperator(
        eltype(species_to_echos),
        nrow,
        ncol, false, false,
        (res,x) -> (res .= forward_model(x, species_to_echos, nx, ny, n_species, n_echos, nfft_plan)),
        nothing,
        (res,x) -> (res .= conj_forward_model(x, species_to_echos, nx, ny, n_species, n_echos, nfft_plan))
    )
end


function get_species_to_echos_mtx(n_species, n_echos, echo_times, weights, df, phi0; relaxation=nothing)
	species_to_echos = zeros(ComplexF32, n_species, n_echos)
	for (i, echo) in enumerate(echo_times)
	    for k=1:length(df)
	        for j=1:length(df[k])
	            species_to_echos[k, i] += weights[k][j]*exp.(df[k][j]*echo*im)
                if relaxation !== nothing
                    species_to_echos[k, i] *= exp(-echo * relaxation[k])
                end
	        end
	        species_to_echos[k, i] *= exp(phi0[k]*im)
	    end
	end
    return species_to_echos
end


function cs_field_inhom_forward_model(x, nx, ny, n_species, n_echos, nfft_operators, cs_mtx)
    x = reshape(x, nx, ny, n_species)

    # field inhomogeniety
    y = Array{eltype(x)}(undef, nx, ny, n_species, n_echos)
    for (species_n, echo_n)=Iterators.product(1:n_species, 1:n_echos)
        y[:, :, species_n, echo_n] = nfft_operators[echo_n] * vec(x[:, :, species_n])
    end

    # cs
    y = reshape(y, nx*ny, n_species*n_echos) * cs_mtx

    return vec(y)
end

function adj_cs_field_inhom_forward_model(y, nx, ny, n_species, n_echos, nfft_operators, cs_mtx)
    x = reshape(y, nx*ny, n_echos)

    # conj cs
    x = x * adjoint(cs_mtx)

    # conj inhomogeniety
    x = reshape(x, nx, ny, n_species, n_echos)
    for (species_n, echo_n)=Iterators.product(1:n_species, 1:n_echos)
        x[:, :, species_n, echo_n] = adjoint(nfft_operators[echo_n]) * vec(x[:, :, species_n, echo_n])
    end

    return vec(sum(x; dims=4))
end

function FieldInhomChemCompOp(nx, ny, n_species, n_echos, fft_operators, cs_mtx)
    nrow = nx*ny*n_echos
    ncol = nx*ny*n_species
    return LinearOperator(
        Float32, #eltype(species_to_echos),
        nrow,
        ncol, false, false,
        (res,x) -> (res .= cs_field_inhom_forward_model(x, nx, ny, n_species, n_echos, fft_operators, cs_mtx)),
        nothing,
        (res,x) -> (res .= adj_cs_field_inhom_forward_model(x, nx, ny, n_species, n_echos, fft_operators, cs_mtx))
    )
end
