params_range = {
    # parameters for snow module
    'param_ts': [-3, 1],  # temperature threshold for snowfall and rainfall, °C.
    'param_tm': [-3, 1],  # temperature threshold for snowmelt, °C.
    'param_ds6': [0., 10],  # degree-day factor for snowmelt on 21 June, mm/°C/day.
    'param_ds12': [0., 10],  # degree-day factor for snowmelt on 21 December, mm/°C/day.
    'param_ds': [0., 10],  # degree-day factor for snowmelt, mm/°C/day.
    'param_snow2ice': [0.0, 0.01],  # a constant coefficient to calculate snow to ice, -.
    'param_rf': [9.3e-25, 2.4e-24],  # rate factor for glacier flow, Pa^-3 s^-1. https://doi.org/10.1017/jog.2018.75
    'param_flow_frac': [0.0, 0.1], # fraction of glacier shifting, -.
    'param_Asp': [0.01, 0.3],  # coefficient to calculate snow pressure, g/cm2.
    'param_beta': [0.85, 1.2],  # exponent coefficient to calculate snow pressure.
    # parameters for glacier module
    'param_tg': [-3, 1],  # temperature threshold for glacier melt, °C.
    'param_dg6': [0., 10],  # day-degree factor for glacier melt on 21 June, mm/°C/day.
    'param_dg12': [0., 10],  # day-degree factor for glacier melt on 21 December, mm/°C/day.
    'param_dg': [0., 10],  # day-degree factor for glacier melt, mm/°C/day.
    'param_m': [0.020, 0.030],  # coefficient for volume-area scaling relationship.
    'param_n': [1.210, 1.240],  # exponent for volume-area scaling relationship.
}