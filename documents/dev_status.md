## Features of PRX

Different information are required depending oh the positioning algorithm chosen. Three types of positioning algorithms have been identified, each requiring specific data that are organized into different levels of processing in PRX.  
The following tables gather the different information needed according to the level of processing.  
    - **Level 1** : suitable for DGNSS/RTK processing.    
    - **Level 2** : designed for Single Point Positioning (SPP).  
    - **Level 3** : designed for Precise Point Positioning (PPP).  

### Level 1 – DGNSS / RTK

Those parameters are computed from the broadcast navigation message (`rinex nav` file).

```
uv run python src/prx/main.py --observation_file_path <path_to_rinex_file> --prx_level 1
```

| Parameters                      | Name in PRX file                                                                                                                                                               | Status    |
|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| GNSS observations               | `time_of_reception_in_receiver_time`<br/>`C_obs_m`, `D_obs_hz`, `L_obs_cycles`, `S_obs_dBHz`,<br/>`rnx_obs_identifier`, `constellation`, `prn`, `frequency_slot` (for GLONASS) | ✅       |
| Loss of Lock indicator          | `LLI`                                                                                                                                                                          | ✅       |
| Satellite health flag           | `health_flag`                                                                                                                                                               | ✅        |
| Satellite position and velocity | `sat_pos_x_m`, `sat_pos_y_m`, `sat_pos_z_m`,<br> `sat_vel_x_mps`, `sat_vel_y_mps`, `sat_vel_z_mps`                                                                             | ✅       |
| Satellite elevation and azimuth | `sat_elevation_deg`, `sat_azimuth_deg`                                                                                                                                         | ✅       |
| Ephemerides dataset identifier  | `ephemeris_hash`                                                                                                                                                               | ✅       |


### Level 2 - SPP 
Same as Level 1, with the following additional parameters. Those parameters are still computed from the broadcast navigation message.

```
uv run python src/prx/main.py --observation_file_path <path_to_rinex_file> --prx_level 2
```

| Parameters                       | Name in PRX file               | Status    |
|----------------------------------|--------------------------------|-----------|
| Satellite clock offset and drift | `sat_clock_offset_m`, `sat_clock_drift_mps`  | ✅       |
| Relativistic clock effect        | `relativistic_clock_effect_m`    | ✅       |
| Sagnac effect                    | `sagnac_effect_m`                | ✅       |
| Tropospheric delay               | `tropo_delay_m`                  | ✅       |
| Satellite code bias              | `sat_code_bias_m`                | ✅       |
| Ionospheric delay                | `iono_delay_m`                   | ✅       |

---

### Level 3 - PPP 
Same as level 1 and 2, with the following additional parameters. The parameters are computed using IGS products.

```
uv run python src/prx/main.py --observation_file_path <path_to_rinex_file> --prx_level 3
```

| Parameters                                                        | Name in PRX file                                                                                  | Status |
|-------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------|
| Satellite position and velocity (computed using `sp3` and `clk` files)                               | `sat_pos_x_m`, `sat_pos_y_m`, `sat_pos_z_m`,<br>`sat_vel_x_mps`, `sat_vel_y_mps`, `sat_vel_z_mps` | ❌      |
| Satellite clock offset and drift (including relativistic effect) (computed using `sp3` and `clk` files) | `sat_clock_offset_m`, `sat_clock_drift_mps`                                                       | ❌      |
| Tropospheric delay (computed using `tropex` files)                                           | `tropo_delay_m`                                                                                   | ❌      |
| Tropospheric mapping function                                     | to be completed                                                                                   | ❌      |
| Ionospheric delay (computed using `ionex` files)                                           | `iono_delay_m`                                                                                    | ❌      |
| Satellite code & phase bias (computed using `bia` files)                                | `sat_code_bias_m`                                                                                 | ❌      |
| Satellite & receiver phase center offset and variation (computed using `antex` files)    | to be completed                                                                                   | ❌      |
| Solid Earth Tide                                                  | to be completed                                                                                   | ❌      |

  

  
