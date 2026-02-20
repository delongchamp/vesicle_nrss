# vesicle_nrss

NRSS morphology and simulation framework for single- and multi-vesicle models.

## Public API

```python
from vesicle_nrss import (
    VesicleArgs,
    VesicleResults,
    default_vesicle_args,
    small_test_args,
    highres_vesicle_args,
    build_vesicle_morph,
    run_vesicle,
    run_vesicle_sweep,
)
```

## Notes

- All dimensional inputs are in nm.
- `oc_lipid` and `oc_medium` are placeholders by default and are expected to be
  provided by notebook/user setup before running NRSS.
- `run_vesicle` returns results only (no default serialization).
- Ray sweeps are cluster-only and require the hardcoded cluster settings from
  the repository spec.
