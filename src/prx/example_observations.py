def generate():
    # create single log as dictionary
    header = {
        "speed_of_light_mps": 299792458.123456,
        "input_files": [
            {"name": "file1.obs", "md5": "9e107d9d372bb6826bd81d3542a419d6"},
            {"name": "file2.nav", "md5": "zjutrd9d372bb6826bd81d3542a47j43"},
        ],
        "reference_frame": "ITRF2008@2005.0",
        "carrier_frequencies": [
            {"constellation": "G", "signal": "L1C", "frequency": "1575420000.000000"}
        ],
        "prx_git_commit_id": "663524c95079f0bf3a91ef0d7efd7abe97f82a76",
    }
    observations = []
    observations.append(
        {
            "tai_s": 1662855733.123456789,
            "constellation": "G",
            "prn": 56,
            "observation_code": "1C",
            "code_observation_m": 20685773.805123,
            "doppler_observation_hz": -1214.816,
            "carrier_observation_m": 20685780.9518393,
            "cn0_dbhz": 48.9,
            "satellite_position_x_m": 13788686.065826,
            "satellite_position_y_m": 6986730.9310050,
            "satellite_position_z_m": 21551023.534684,
            "satellite_velocity_x_mps": -601.654533296824,
            "satellite_velocity_y_mps": 2681.16549309343,
            "satellite_velocity_z_mps": -473.157677799463,
            "satellite_clock_bias_m": -42073.746123,
            "satellite_clock_bias_drift_mps": -0.003238,
            "group_delay_s": 0.558407,
            "sagnac_effect_m": -7.463553,
            "tropo_delay_m": 2.612978,
            "iono_delay_m": 1.64424645061389,
            "constellation_time_minus_tai_s": 19,
            "approximate_antenna_position_x_m": 4627869.62515177,
            "approximate_antenna_position_y_m": 119635.618137254,
            "approximate_antenna_position_z_m": 4373009.98713997,
        }
    )
    observations.append(observations[0].copy())
    observations[1]["observation_code"] = "2C"
    return header, observations
