# Athena vs neural-fortran benchmarks

Matched timing comparisons between Athena and neural-fortran. Each pair uses the
same architecture, data and iteration count; loss and learning-rate scaling are
chosen to match effective gradients. Timing covers only the training loop;
network construction, data setup and final evaluation are excluded.

## MNIST

One-epoch timing benchmark on identical data and architecture.

- Data: the same plain-text MNIST files used by the Athena example
  (`MNIST_train.txt`, `MNIST_test.txt`), first 50,000 rows for training,
  10,000 test rows, pixels scaled to [0,1].
- Architecture: `conv2d(32, 3x3, relu, valid)` -> `maxpool2d(2,2)` ->
  `dense(100, relu)` -> `dense(10, softmax)`.
- Training: MSE loss, SGD without momentum, learning rate 0.01, batch 32,
  one epoch (1,562 steps in both libraries), fixed seed.
- Timing: wall clock around the single `train` call; data loading and the final
  test pass are excluded.

## Simple function

Adapted from the `simple` example in both libraries: learn
`y = [0.123456, 0.246802]` from `x = [0.2, 0.4, 0.6]`.

- Architecture: `input(3)` -> `dense(5, tanh)` -> `dense(2, sigmoid)`.
- Loss and optimizer: MSE in both apps, SGD with batch 1. Athena uses learning
  rate 2 and neural-fortran uses learning rate 1: Athena's MSE is mean/2 while
  neural-fortran's is sum/n, so the doubled Athena rate matches the effective
  gradients.
- Training: 501 iterations (`n = 0..500`), one pass, fixed seed
  (the `random_seed` array is filled with ones).
- Timing: wall clock from `wall_time()` (POSIX
  `clock_gettime(CLOCK_MONOTONIC)`, nanosecond resolution) and CPU time around
  the single pass; network construction and final evaluation are excluded.
- Metric: prediction MSE on the training sample after the single pass.

## Sine function

Adapted from the `sine` example in both libraries: approximate
`(sin(x) + 1) / 2` over `x` in `[0, 2*pi]`.

- Architecture: `input(1)` -> `dense(5, tanh)` -> `dense(1, sigmoid)`.
- Loss and optimizer: MSE in both apps, SGD with batch 1. Athena uses learning
  rate 2 and neural-fortran uses learning rate 1 for the same reason as the
  simple benchmark (mean/2 versus sum/n MSE).
- Training: each iteration draws a random `x` in `[0, 2*pi]` and sets
  `y = (sin(x) + 1) / 2`; 100,001 iterations (`n = 0..100000`), fixed seed.
  Both apps re-seed immediately before the training loop, after network
  construction, so the sampled `x` sequence is identical.
- Timing: nanosecond wall clock (`wall_time()`) around the training loop only.
- Metric: MSE on 30 evenly spaced test points
  `x = (i - 1) * 2*pi / 30`, `y = (sin(x) + 1) / 2`, after training.

## Deliberate deviations from the original examples

- The hidden layer uses tanh in both apps; neural-fortran's `dense` defaults to
  sigmoid.
- The sine benchmark uses 100,001 iterations; the Athena example uses 10,001.
- The neural-fortran SGD optimizer is constructed once outside the loop
  instead of once per iteration.
- The simple benchmark times a single 501-step pass with a nanosecond POSIX
  wall clock; no repeated passes are used.
- Athena uses learning rate 2 against neural-fortran's 1 to compensate for
  Athena's mean/2 MSE versus neural-fortran's sum/n MSE.
- Both Athena loops release the consumed loss graph each step
  (`loss%release_expression()`), as `network%train` does internally. Without
  this cleanup the manual 100,000-step sine loop accumulates expression history
  and takes roughly 151 s.

## Manual-loop memory retention

An earlier version retained the loss-root gradient record for every manual
step: `release_expression` discarded the root operands before checking whether
the root was a variable, so the root looked like a variable and its gradient
was zeroed instead of released. Releasing that gradient before discarding the
operands keeps the manual 100,001-step sine loop at about 4 MB peak RSS instead
of growing to 97 MB, and removes the cumulative slowdown (about 0.43 s instead
of 8.8 s).

## Run

```sh
MNIST_DIR=/path/to/MNIST-txt ./run.sh
```

The script runs the simple and sine pairs first, then runs the MNIST pair only
if `$MNIST_DIR/MNIST_train.txt` exists; otherwise it prints a skip message.
`MNIST_DIR` defaults to `../../../MNIST-txt` relative to this directory. The
script builds/runs the four simple/sine executables, plus the two MNIST
executables when MNIST data are present. The first run fetches neural-fortran
(the branch specified in `fpm.toml`) and builds both libraries.

To run a single benchmark directly:

```sh
fpm run benchmark_sine_athena --profile release \
  --flag "-O3 -fopenmp" --link-flag "-fopenmp"
```

## Observed (Apple M2 Pro, gfortran 15.2)

Observed with neural-fortran revision
`17d01b55582e5df19557d2d70e03211f9b1820d9` (from `build/cache.toml`).
`fpm.toml` follows the `main` branch, so results may change with upstream.

### MNIST

| library | train (s) | test accuracy | peak RSS |
|---|---|---|---|
| Athena | 5.596 | 0.7791 | 810.7 MB |
| neural-fortran | 58.10 | 0.3468 | 202 MB |

Athena's test loss for this run was 0.02079. Accuracy is after a single epoch
and is not a convergence comparison; the libraries use different
initialisation, batch sampling (Athena shuffles a full pass, neural-fortran
samples batches with replacement) and update implementations. The timing covers
the same number of samples, layers, batch size and optimiser steps. Peak RSS
was measured with `/usr/bin/time -l`.

Regression controls (separate from the matched benchmarks): the Athena MNIST
example, run for one epoch, trained in 15.34 s with accuracy 0.98140, loss
0.05878 and 713 MB peak RSS; the PINN Burgers example reached loss 0.21799
after 100 epochs with no NaNs. These controls are not matched-library
comparisons.

### Simple function

| library | wall (s) | CPU (s) | final MSE |
|---|---|---|---|
| Athena | 0.00295 | 0.00277 | 2.78e-15 |
| neural-fortran | 0.000862 | 0.000864 | 1.30e-13 |

One 501-step pass, timed with the nanosecond wall clock. Wall times varied
between processes; the table shows the latest full run, while warm repeated
runs after the release-order fix were about 2.81–3.20 ms (Athena) and 0.30 ms
(neural-fortran). Both networks fit the two targets to near machine precision.

### Sine function

| library | train wall (s) | train CPU (s) | test MSE |
|---|---|---|---|
| Athena | 0.425 | 0.425 | 3.73e-03 |
| neural-fortran | 0.0654 | 0.0654 | 4.21e-03 |

Athena's peak RSS for this run was 3.7 MB. The final MSE depends on
initialisation and is not a convergence claim. The timed work is the same
number of samples, layers, batch size and effective optimiser steps.
