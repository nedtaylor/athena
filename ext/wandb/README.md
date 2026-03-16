# athena wandb extension

A Fortran interface to [Weights & Biases](https://wandb.ai) (wandb)
experiment tracking, built on the Python C API.
This extension to athena that enables wandb capabilities has been developed with the help of AI agents.

## Prerequisites

- Python >= 3.8 with development headers
  - macOS: `brew install python3` (or Xcode Command Line Tools)
  - Linux: `sudo apt install python3-dev` (Debian/Ubuntu)
- The `wandb` Python package: `pip install wandb`
- A wandb account and API key (run `wandb login` once)

The code has been tested using Python == 3.12, fpm == 0.12.0, and wandb 0.25.0.

## Building

For building and testing using fpm, follow this from the athena project root (assumes you already have a wandb account and have logged in):

```bash
# Set the correct Fortran Python version
PYTHON=PATH_TO_YOUR_PYTHON source tools/setup_wandb_python_env.sh
fpm run wandb_sine --example
```


For building using CMake, follow this:

From the athena project root:

```bash
# Build the wandb extension
cmake -S ext/wandb -B build_wandb
cmake --build build_wandb
```

## API

```fortran
use athena_wandb

! Start a run
call wandb_init(project="my_project", name="run-01")

! Log hyper-parameters
call wandb_config_set("learning_rate", 0.001)
call wandb_config_set("num_hidden", 64)
call wandb_config_set("activation", "relu")

! Log metrics during training
call wandb_log("loss", loss_value, step=epoch)
call wandb_log("accuracy", acc_value, step=epoch)

! Finish the run
call wandb_finish()
```

### Routines

| Routine | Description |
|---|---|
| `wandb_init(project [, name] [, entity])` | Initialise a wandb run |
| `wandb_log(key, value [, step])` | Log a scalar metric (real32, real64, or integer) |
| `wandb_config_set(key, value)` | Set a config parameter (integer, real32, real64, or string) |
| `wandb_finish()` | Finish the run and release resources |

## Linking in your project

After building, link against `libathena_wandb.a` and Python:

```cmake
add_subdirectory(ext/wandb)
target_link_libraries(my_app PRIVATE athena_wandb)
```

Or with a Makefile:

```makefile
WANDB_DIR = path/to/build_wandb
WANDB_FLAGS = -I$(WANDB_DIR)/mod -L$(WANDB_DIR) -lathena_wandb $(shell python3-config --ldflags --embed)
```
