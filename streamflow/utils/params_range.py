params_range = {
'glac_params' : {
    # parameters for snow module
    'param_ts_glac': [-3, 1],  # temperature threshold for snowfall and rainfall, °C.
    'param_tm_glac': [-3, 1],  # temperature threshold for snowmelt, °C.
    'param_ds6_glac': [0, 10],  # degree-day factor for snowmelt on 21 June, mm/°C/day.
    'param_ds12_glac': [0, 10],  # degree-day factor for snowmelt on 21 December, mm/°C/day.
    'param_ds_glac': [0, 10],  # degree-day factor for snowmelt, mm/°C/day.
    'param_snow2ice_glac': [0.0, 0.01],  # a constant coefficient to calculate snow to ice, -.
    'param_rf_glac': [9.3e-25, 2.4e-24],  # rate factor for glacier flow, Pa^-3 s^-1. https://doi.org/10.1017/jog.2018.75
    'param_flow_frac_glac': [0.0, 0.1], # fraction of glacier shifting, -.
    'param_Asp_glac': [0.01, 0.3],  # coefficient to calculate snow pressure, g/cm2.
    'param_beta_glac': [0.85, 1.2],  # exponent coefficient to calculate snow pressure.
    # parameters for glacier module
    'param_tg_glac': [-3, 1],  # temperature threshold for glacier melt, °C.
    'param_dg6_glac': [0, 10],  # day-degree factor for glacier melt on 21 June, mm/°C/day.
    'param_dg12_glac': [0, 10],  # day-degree factor for glacier melt on 21 December, mm/°C/day.
    'param_dg_glac': [0, 10],  # day-degree factor for glacier melt, mm/°C/day.
    'param_m_glac': [0.020, 0.030],  # coefficient for volume-area scaling relationship.
    'param_n_glac': [1.210, 1.240],  # exponent for volume-area scaling relationship.
},

'rr_params' : {
    'param_tm_rr': [-3, 1],  # temperature threshold for snowmelt, °C.
    'param_ds_rr': [0, 10],  # degree-day factor for snowmelt, mm/°C/day.
    'param_ds6_rr': [0, 10],  # degree-day factor for snowmelt on 6 June, mm/°C/day.
    'param_ds12_rr': [0, 10],  # degree-day factor for snowmelt on 21 December, mm/°C/day.
    'param_ts_rr': [-3, 1],  # temperature threshold for snowfall and rainfall.
    'param_snow2ice_rr': [0.0, 0.01],  # coefficient to calculate snow to ice conversion.
    'param_Asp_rr': [0.01, 0.3],  # coefficient to calculate snow pressure, g/cm2.
    'param_beta_rr': [0.85, 1.2],  # exponent coefficient to calculate snow pressure.
    'param_Imax_rr': [0.1, 5],  # maximum interception storage, mm.
    'param_uWUE_rr': [0.001, 0.11],  # water use efficiency, kg CO2/kg H2O. 10.1029/2022WR033978/ 10.1029/2024MS004308
    'param_Ksg_rr': [0.0, 0.12],  # natural decay factor for green biomass, d-1.
    'param_Cg_rr': [0.005, 0.05],  # coefficient to calculate LAI from green biomass
    'param_slp_rr': [5, 15],  # parameter to calculate carbon allocation to green biomass, controls stepness of curve.
    'param_lmt_rr': [0.5, 1.0],  # natural decay factor for brown biomass, d-1, controls the threshold of brown biomass.
    'param_Smax_rr': [100, 1500],  # maximum storage of the catchment bucket, mm.
    'param_S0_rr': [0, 1500], # initial storage of the catchment bucket, mm.
    'param_F_rr': [0, 0.1],  # rate of decline in flow from catchment bucket.
    'param_Qmax_rr': [10, 50],  # maximum subsurface flow at full bucket, mm/day.
    'param_freCoef_rr': [0, 1],  # a parameter determining how much soil water is freezing.
    'param_Ks_rr': [0.5, 1],  # soil coefficient to calculate the maximum bare soil evaporation from ET0.
    'param_A_rr': [0, 2.9],  # shape parameter for gamma distribution.
    'param_B_rr': [0, 6.5],  # timescale parameter for gamma distribution.
    'param_X_rr': [0.01, 0.5],  # one of muskingum parameters, controls the ratio of inflow to outflow.
    'param_K_rr': [1, 10],  # one of muskingum parameters, measures the time delay of the flow.
}
}