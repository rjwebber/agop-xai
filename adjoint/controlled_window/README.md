# Controlled-window composition

`zc_one_step_driver.F` advances one complete packed Zebiak--Cane state by one
coupled ten-day step. `zc_one_step_tangent_driver.F` advances the same state
and a 59,148-coordinate real tangent together. They link against the immutable
primal and generated tangent objects from the curated run23 final3 build; they
do not change the canonical adjoint evidence.

The driver exists so an external optimization can insert an explicitly bounded
native-state increment between steps, then let the authentic model propagate
that state.  This is the forward primitive for gradual covariance controls.
The corresponding linearized recurrence uses the already generated one-step
Tapenade tangent and reverse kernels.

## Python composition bridge

`src/zc_xai/zc_controlled_bridge.py` wraps these executables without changing
the certified build.  Every operation copies the runtime to a private
temporary directory and uses temporary state, tangent, and tape files.  Its
three main operations are:

- `build_zero_control_path(...)`, which retains every unforced boundary state
  and one three-integer branch tape per step;
- `form_authentic_release_response(...)`, which composes the one-step tangent,
  a constant or coherently phase-local packed control map, and the staggered
  training-standardized core4 observation; and
- `replay_control(...)`, which inserts a distributed increment immediately
  before each controlled step and then advances the authentic nonlinear model.

The release-response builder requires the tangent routine's simultaneous
primal replay to satisfy the final3 numerical tolerances and requires exact
agreement of every stored branch tape.  `verify_zero_control_replay(...)`
separately requires bitwise equality of every state and tape for the explicit
zero-dose safety case.

This tangent composition supported the reduced pilot calculations. The final
nonlinear action-minimization experiment uses the matched one-step reverse
bridge in `adjoint/controlled_reverse/`, because repeated SQP gradients with
respect to nine native-state perturbations are substantially cheaper through
one backward sweep than through a coordinate-by-coordinate tangent basis.

Build locally with:

```bash
bash adjoint/controlled_window/build_one_step_driver.sh
```

Run inside a validated ZC runtime directory containing `fc.data`, the restart,
namelists, and `Data/`:

```bash
adjoint/controlled_window/build/zc_one_step input_state.bin output_state.bin tape.bin
```

The executable supports only the deterministic Standard configuration already
claimed by run23.  A controlled trajectory remains subject to the branchwise,
single-precision qualifications in `docs/ZC_ADJOINT_DESIGN.md`.
