TD3_HalfCheetah-v5_0_1:

```
    param_ranges = {
        "gravity_scale": [0.8, 1.2],
        "friction_scale": [0.7, 1.3],
        "mass_scale": [0.9, 1.1],
    }
    seed = 0
    def eval_policy():
        ...
        mu = np.array([1.0, 1.0, 1.0])
        ...
```

TD3_HalfCheetah-v5_0_2:
```
    param_ranges = {
        "gravity_scale": [0.7, 1.3],
        "friction_scale": [0.5, 1.5],
        "mass_scale": [0.8, 1.2],
    }
    seed = 0
        def eval_policy():
        ...
        mu = np.array([1.0, 1.0, 1.0])
        ...
```
