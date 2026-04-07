#!/usr/bin/env bash
# tools/setup_wf_env.sh — export Python flags so that fpm can compile and link
# the Fortran examples that use the C API of the wandb Python package.
# This file is copied from the wandb-fortran repository to simplify setting up the
# Python environment for the examples in this repository.
#
# SOURCE this file — do not execute it:
#   source tools/setup_env.sh
#   PYTHON=/path/to/python source tools/setup_env.sh
#
# After sourcing you can run fpm directly:
#   fpm build
#   fpm run --example athena_logging
#
# To make this permanent, add the source line to your ~/.zshrc or ~/.bashrc.
#
# Prerequisites for the chosen Python:
#   <python> -m pip install wandb && wandb login

# Use return (not exit) so the file is safe to source from interactive shells.
_wf_err() { echo "ERROR (setup_env.sh): $*" >&2; return 1 2>/dev/null || exit 1; }

_wf_save_original_env() {
    if [[ -z "${_WF_ORIG_FPM_CFLAGS+x}" ]]; then
        export _WF_ORIG_FPM_CFLAGS="${FPM_CFLAGS:-}"
    fi
    if [[ -z "${_WF_ORIG_FPM_LDFLAGS+x}" ]]; then
        export _WF_ORIG_FPM_LDFLAGS="${FPM_LDFLAGS:-}"
    fi
    if [[ -z "${_WF_ORIG_DYLD_LIBRARY_PATH+x}" ]]; then
        export _WF_ORIG_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"
    fi
}

_wf_python_has_wandb() {
    local candidate="$1"
    [[ -n "$candidate" ]] || return 1
    command -v "$candidate" &>/dev/null || return 1
    "$candidate" -c "import wandb, wandb.errors" >/dev/null 2>&1
}

# --------------------------------------------------------------------------- #
#  Resolve Python interpreter                                                  #
# --------------------------------------------------------------------------- #
if [[ -n "${PYTHON:-}" ]]; then
    _WF_PY="${PYTHON}"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    _WF_PY="${CONDA_PREFIX}/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    _WF_PY="${VIRTUAL_ENV}/bin/python"
elif command -v python &>/dev/null; then
    _WF_PY="python"
else
    _WF_PY="python3"
fi

if ! command -v "$_WF_PY" &>/dev/null; then
    _wf_err "Python interpreter '$_WF_PY' not found."
fi

if [[ -z "${PYTHON:-}" ]] && ! _wf_python_has_wandb "$_WF_PY"; then
    if _wf_python_has_wandb python3; then
        _WF_PY="python3"
    elif command -v conda &>/dev/null; then
        _WF_CONDA_BASE=$(conda info --base 2>/dev/null)
        if [[ -n "$_WF_CONDA_BASE" && -d "$_WF_CONDA_BASE/envs" ]]; then
            for _WF_CONDA_PY in "$_WF_CONDA_BASE"/envs/*/bin/python; do
                [[ -x "$_WF_CONDA_PY" ]] || continue
                if _wf_python_has_wandb "$_WF_CONDA_PY"; then
                    _WF_PY="$_WF_CONDA_PY"
                    break
                fi
            done
        fi
    fi
fi

_WF_PY_ABS=$(command -v "$_WF_PY")
_WF_PY_VER=$("$_WF_PY_ABS" -c \
    "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
_WF_PY_DIR=$(dirname "$_WF_PY_ABS")
_WF_PY_PREFIX=$("$_WF_PY_ABS" -c "import sys; print(sys.prefix)")
_WF_PY_PURELIB=$("$_WF_PY_ABS" -c "import sysconfig; print(sysconfig.get_path('purelib') or '')")
_WF_PY_PLATLIB=$("$_WF_PY_ABS" -c "import sysconfig; print(sysconfig.get_path('platlib') or '')")

# Prefer the versioned python-config alongside the interpreter.
_WF_PY_CFG="${_WF_PY_DIR}/python${_WF_PY_VER}-config"
[[ -x "$_WF_PY_CFG" ]] || _WF_PY_CFG="${_WF_PY_DIR}/python3-config"

if [[ ! -x "$_WF_PY_CFG" ]]; then
    _wf_err "Could not find python${_WF_PY_VER}-config or python3-config \
alongside '$_WF_PY_ABS'. Install Python development headers."
fi

# --------------------------------------------------------------------------- #
#  Derive flags                                                                #
# --------------------------------------------------------------------------- #
_wf_save_original_env

_WF_INCLUDES=$("$_WF_PY_CFG" --includes)
_WF_LDFLAGS=$("$_WF_PY_CFG" --ldflags --embed 2>/dev/null \
              || "$_WF_PY_CFG" --ldflags)

# python-config's -L often points only to config-X.Y-darwin/, not the directory
# that holds the actual libpython dylib. Prepend the real LIBDIR and bake in
# an rpath so the binary finds the dylib without DYLD_LIBRARY_PATH at runtime.
_WF_LIBDIR=$("$_WF_PY_ABS" -c \
    "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
if [[ -n "$_WF_LIBDIR" ]]; then
    _WF_LDFLAGS="-L${_WF_LIBDIR} -Wl,-rpath,${_WF_LIBDIR} ${_WF_LDFLAGS}"
    export DYLD_LIBRARY_PATH="${_WF_LIBDIR}${_WF_ORIG_DYLD_LIBRARY_PATH:+:${_WF_ORIG_DYLD_LIBRARY_PATH}}"
fi

_WF_PY_PATH="$_WF_PY_PURELIB"
if [[ -n "$_WF_PY_PLATLIB" && "$_WF_PY_PLATLIB" != "$_WF_PY_PURELIB" ]]; then
    _WF_PY_PATH="${_WF_PY_PATH:+${_WF_PY_PATH}:}${_WF_PY_PLATLIB}"
fi

# --------------------------------------------------------------------------- #
#  Export                                                                      #
# --------------------------------------------------------------------------- #
export FPM_CFLAGS="${_WF_ORIG_FPM_CFLAGS}${_WF_ORIG_FPM_CFLAGS:+ }${_WF_INCLUDES}"
export FPM_LDFLAGS="${_WF_ORIG_FPM_LDFLAGS}${_WF_ORIG_FPM_LDFLAGS:+ }${_WF_LDFLAGS}"

# Filter out -framework flags which are not supported by flang.
export FPM_LDFLAGS=$(echo "$FPM_LDFLAGS" | sed 's/ -framework [^ ]*//g')

export WF_PYTHON_BIN="$_WF_PY_ABS"
export WF_PYTHON_HOME="$_WF_PY_PREFIX"
export WF_PYTHON_PATH="$_WF_PY_PATH"

echo "setup_env.sh: configured fpm for Python ${_WF_PY_VER} (${_WF_PY_ABS})"
echo "  FPM_CFLAGS  = $FPM_CFLAGS"
echo "  FPM_LDFLAGS = $FPM_LDFLAGS"
echo "  WF_PYTHON_HOME = $WF_PYTHON_HOME"
echo "  WF_PYTHON_PATH = $WF_PYTHON_PATH"
echo ""
echo "You can now run:  fpm build"

# --------------------------------------------------------------------------- #
#  Tidy up — don't leak private variables into the caller's shell              #
# --------------------------------------------------------------------------- #
unset _WF_PY _WF_PY_ABS _WF_PY_VER _WF_PY_DIR
unset _WF_PY_CFG _WF_PY_PREFIX _WF_PY_PURELIB _WF_PY_PLATLIB _WF_PY_PATH
unset _WF_INCLUDES _WF_LDFLAGS _WF_LIBDIR
unset _WF_CONDA_BASE _WF_CONDA_PY
unset -f _wf_save_original_env
unset -f _wf_python_has_wandb
