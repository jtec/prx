## Features of PRX

Different information are required dpending oh the positioning algorithm chosen. Three types of positioning algorithms have been identified, each requiring specific data that are organized into different levels of processing in PRX.  
The following tables gather the different information needed according to the level of processing.  
    - **Level 1** : suitable for DGNSS/RTK processing.    
    - **Level 2** : designed for Single Point Positioning (SPP).  
    - **Level 3** : designed for Precise Point Positioning (PPP).  

PRX parameters not used for the 3 levels of positioning algorithms : code_id, ephemeris_hash, frequency_slot, carrier_frequency_hz


### Level 1 – DGNSS / RTK

| Parameters                        | PRX file                       | Status    |
|-----------------------------------|--------------------------------|-----------|
| GNSS observations                 | time_of_reception_in_receiver_time, C_obs_m, D_obs_hz,    L_obs_cycles, S_obs_dBHz, rnx_obs_identifier, constellation, prn                               | ✅       |
| Loss of Lock indicator            | LLI                            | ✅       |
| Satellite health flag             |                                | ❌       |
| Satellite position and velocity   |   - sat_pos_x_m, sat_pos_y_m, sat_pos_z_m,<br>- sat_vel_x_mps, sat_vel_y_mps, sat_vel_z_mps                                                        | ✅       |

--- 

### Level 2 - SPP 

| Parameters                        | PRX file                       | Status    |
|-----------------------------------|--------------------------------|-----------|
| GNSS observations                 |  time_of_reception_in_receiver_time, C_obs_m, D_obs_hz,    L_obs_cycles, S_obs_dBHz, rnx_obs_identifier, prn                                              | ✅       |
| Loss of Lock indicator            | LLI                            | ✅       |
| Satellite health flag             |                                | ❌       |
| Satellite position and velocity   |  - sat_pos_x_m, sat_pos_y_m, sat_pos_z_m,<br>- sat_vel_x_mps, sat_vel_y_mps, sat_vel_z_mps                                                        | ✅       |
| Satellite clock offset and drift  | sat_clock_offset_m, sat_clock_drift_mps  | ✅       |
| Relativistic clock effect         | relativistic_clock_effect_m    | ✅       |
| Sagnac effect                     | sagnac_effect_m                | ✅       |
| Tropospheric delay                | tropo_delay_m                  | ✅       |
| Satellite code bias               | sat_code_bias_m                | ✅       |
| Ionospheric delay                 | iono_delay_m                   | ✅       |
| Satellite elevation and azimut    | sat_elevation_deg, sat_azimuth_deg       | ✅       |

---

### Level 3 - PPP 

| Parameters                        | PRX file                       | Status    |
|-----------------------------------|--------------------------------|-----------|
| GNSS observations                 | time_of_reception_in_receiver_time, C_obs_m, D_obs_hz,    L_obs_cycles, S_obs_dBHz, rnx_obs_identifier , prn                                             | ✅       |
| Loss of Lock indicator            |   LLI                          | ✅       |
| Satellite health flag             |                                | ❌       |
| Satellite position and velocity   | - sat_pos_x_m, sat_pos_y_m, sat_pos_z_m,<br>- sat_vel_x_mps, sat_vel_y_mps, sat_vel_z_mps                                                        | ✅       |
| Satellite clock offset and drift  | sat_clock_offset_m, sat_clock_drift_mps | ✅       |
| Relativistic clock effect         | relativistic_clock_effect_m    | ✅       |
| Sagnac effect                     | sagnac_effect_m                | ✅       |
| Tropospheric delay                | tropo_delay_m                  | ✅       |
| Tropospheric mapping function     |                                | ❌       |
| Satellite code & phase bias       | sat_code_bias_m                | 🟨       |
| Satellite & receiver phase center offset and variation  |          | ❌       |
| Ionospheric delay                 | iono_delay_m                   | ✅       |
| Solid Earth Tide                  |                                | ❌       |
| Satellite elevation and azimut    | sat_elevation_deg, sat_azimuth_deg      | ✅       |
 
