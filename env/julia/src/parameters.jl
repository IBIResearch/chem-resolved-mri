module Parameters
    rho_water = 1.0
	rho_acetone = 0.7845
	rho_ethanol = 0.78945
	molar_mass_water = 18
	molar_mass_acetone = 58.08
	molar_mass_ethanol = 46.068
	nh_water = 2
	nh_acetone = 6
	nh_ethanol = 6
    M0_water = rho_water/molar_mass_water*nh_water
	M0_acetone = rho_acetone/molar_mass_acetone*nh_acetone
	M0_ethanol = rho_ethanol/molar_mass_ethanol*nh_ethanol

    function acetone_frequency_correlation(expected_molar_ratio)
        return - (2.11 - 0.97 * expected_molar_ratio)
    end

	ethanol_frequencies = [-392.6, -82, 143.6] .* 2pi ./ 1000

	ethanol_normalized_weights = [
		0.8976900461254684
 		0.3419771604287499
 		0.27785644284835925
	]

	measured_aceton_frequency_ethanol_experiment = -211 * 2pi / 1000
end