TD3_HalfCheetah-v5_0_dr_1:
```
    param_ranges = {
        "gravity_scale": [0.7, 1.3],
        "friction_scale": [0.5, 1.5],
        "mass_scale": [0.8, 1.2],
    }
    seed = 0
```

TD3_HalfCheetah-v5_0_dr_2:
```
    param_ranges = {
        "gravity_scale": [0.5, 1.5],
        "friction_scale": [0.5, 2.5],
        "mass_scale": [0.6, 1.4],
    }
    seed = 0
```

TD3_HalfCheetah-v5_0_dr_4:
```
    param_ranges = {
        "gravity_scale": [0.8, 1.2],
        "friction_scale": [0.7, 1.3],
        "mass_scale": [0.9, 1.1],
    }
    seed = 0
```

TD3_HalfCheetah-v5_0_dr_3:
```
    param_ranges = {
        "gravity_scale": [0.6, 1.4],
        "friction_scale": [0.5, 2.0],
        "mass_scale": [0.7, 1.5],
    }
    seed = 0
```
忘记改attempt导致把dr_3覆盖了...只好把3,4调换一下，重新训练4