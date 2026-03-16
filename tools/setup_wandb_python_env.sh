#!/usr/bin/env bash
# setup_env.sh — export Python flags so that fpm can compile and link
# wandb_c.c (which embeds Python) without any wrapper script.
#
# This file has been generated using LLM AI code agents and has not yet been thoroughly tested.
#
# SOURCE this file — do not execute it:
#   source example/wandb_sine/setup_env.sh
#   PYTHON=/path/to/python3 source example/wandb_sine/setup_env.sh
#
# After sourcing you can run fpm directly:
#   fpm run wandb_sine --example
#
# To make this permanent, add the source line to your ~/.zshrc or ~/.bashrc.
#
# Prerequisites for the chosen Python:
#   <python> -m pip install wandb && wandb login

# Use return (not exit) so the file is safe to source from interactive shells.
_wandb_err() { echo "ERROR (setup_env.sh): $*" >&2; return 1 2>/dev/null || exit 1; }

# --------------------------------------------------------------------------- #
#  Resolve Python interpreter                                                  #
# --------------------------------------------------------------------------- #
_WANDB_PY="${PYTHON:-python3}"

if ! command -v "$_WANDB_PY" &>/dev/null; then
    _wandb_err "Python interpreter '$_WANDB_PY' not found."
fi

_WANDB_PY_ABS=$(command -v "$_WANDB_PY")
_WANDB_PY_VER=$("$_WANDB_PY_ABS" -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
_WANDB_PY_DIR=$(dirname "$_WANDB_PY_ABS")

# Prefer the versioned python-config (python3.12-config) alongside the interpreter.
_WANDB_PY_CFG="${_WANDB_PY_DIR}/python${_WANDB_PY_VER}-config"
[[ -x "$_WANDB_PY_CFG" ]] || _WANDB_PY_CFG="${_WANDB_PY_DIR}/python3-config"

if [[ ! -x "$_WANDB_PY_CFG" ]]; then
    _wandb_err "Could not find python${_WANDB_PY_VER}-config or python3-config \
alongside '$_WANDB_PY_ABS'. Install Python development headers."
fi

# --------------------------------------------------------------------------- #
#  Derive flags                                                                #
# --------------------------------------------------------------------------- #
_WANDB_INCLUDES=$("$_WANDB_PY_CFG" --includes)
_WANDB_LDFLAGS=$("$_WANDB_PY_CFG" --ldflags --embed 2>/dev/null \
                 || "$_WANDB_PY_CFG" --ldflags)

# python-config's -L often points only to config-X.Y-darwin/, not the directory
# that holds the actual libpython dylib.  Prepend the real LIBDIR and bake in
# an rpath so the binary finds the dylib without DYLD_LIBRARY_PATH at runtime.
_WANDB_LIBDIR=$("$_WANDB_PY_ABS" -c \
    "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
if [[ -n "$_WANDB_LIBDIR" ]]; then
    _WANDB_LDFLAGS="-L${_WANDB_LIBDIR} -Wl,-rpath,${_WANDB_LIBDIR} ${_WANDB_LDFLAGS}"
    # Also keep DYLD_LIBRARY_PATH so existing cached binaries can find the dylib
    # without a forced relink.
    export DYLD_LIBRARY_PATH="${_WANDB_LIBDIR}${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

# --------------------------------------------------------------------------- #
#  Export                                                                      #
# --------------------------------------------------------------------------- #
export FPM_CFLAGS="${FPM_CFLAGS:-} ${_WANDB_INCLUDES}"
export FPM_LDFLAGS="${FPM_LDFLAGS:-} ${_WANDB_LDFLAGS}"

echo "setup_env.sh: configured fpm for Python ${_WANDB_PY_VER} (${_WANDB_PY_ABS})"
echo "  FPM_CFLAGS  = $FPM_CFLAGS"
echo "  FPM_LDFLAGS = $FPM_LDFLAGS"
echo ""
echo "You can now run:  fpm run wandb_sine --example"

# --------------------------------------------------------------------------- #
#  Tidy up — don't leak private variables into the caller's shell              #
# --------------------------------------------------------------------------- #
unset _WANDB_PY _WANDB_PY_ABS _WANDB_PY_VER _WANDB_PY_DIR
unset _WANDB_PY_CFG _WANDB_INCLUDES _WANDB_LDFLAGS _WANDB_LIBDIR
unset -f _wandb_err
