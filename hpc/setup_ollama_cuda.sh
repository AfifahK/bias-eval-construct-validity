#!/bin/bash
# Sourced AT RUNTIME on the GPU node by run_step1_generate.sh / run_step3_judges.sh
# (and by the decider / autoselect smoke tests).
#
# Problem: ollama_v0305 bundles a CUDA 12.8 stack (libcublas/libcublasLt/libcudart)
# that, on SOME ampere EPYC Zen3 nodes, SIGILLs on load and on others silently runs
# the model on CPU (3-5x slower). The combo that survives AND uses the GPU here is a
# full consistent triplet swap to the system cuda module libs (OLLAMA_FORCE_SYSTEM=1,
# proven 2026-06-10 winner=system121, gemma3:1b 12s on gpu-q-76). But it is partly
# NODE-SPECIFIC, so callers should verify on-node and fall back (see
# ollama_serve_resilient.sh, which cycles configs until one drives the GPU).
#
# This file: selects the cuda module + defines ollama_apply_cuda() (re-points the
# bundled cuda_v12 .so.12 symlinks for a chosen config) and applies the winner once
# at source time. Idempotent + reversible (only re-points symlinks; bundled
# .so.<ver> originals stay in place).
#
# Config knobs (env): OLLAMA_FORCE_SYSTEM=1 (system module libs), OLLAMA_BUNDLED=1
# (original bundled 12.8, no swap), OLLAMA_VENDOR_LIB=<dir> (vendored triplet),
# OLLAMA_EXTRA_MODULE=cuda/X.Y (pin module). Exports: OLLAMA_BIN, LD_LIBRARY_PATH,
# OLLAMA_MODELS. `exit 1` (fatal) on unrecoverable setup failure.

: "${OLLAMA_DIR:=/home/$USER/rds/hpc-work/ollama_v0305}"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
[ -x "$OLLAMA_BIN" ] || { echo "[ollama-cuda][FATAL] ollama binary not found/executable: $OLLAMA_BIN"; exit 1; }

# Honor a proven winner config unless caller is iterating (OLLAMA_NO_WINNER=1).
WINNER_ENV="/home/$USER/rds/hpc-work/hpc_run_extension/logs/CUDA_WINNER.env"
if [ "${OLLAMA_NO_WINNER:-0}" != 1 ] && [ -f "$WINNER_ENV" ]; then
    source "$WINNER_ENV"; echo "[ollama-cuda] honoring winner config from $WINNER_ENV"
fi

# Highest available system cuda/12.x module on THIS node (self-heals when CSD3
# adds/removes versions). Override with OLLAMA_EXTRA_MODULE. "__none__" = skip.
CUDA_MOD="${OLLAMA_EXTRA_MODULE:-}"
if [ "$CUDA_MOD" = "__none__" ]; then
    CUDA_MOD=""; echo "[ollama-cuda] cuda module load skipped (OLLAMA_EXTRA_MODULE=__none__)"
else
    [ -z "$CUDA_MOD" ] && CUDA_MOD=$(module avail cuda 2>&1 | tr ' \t' '\n\n' | grep -oE 'cuda/12\.[0-9]+' | sort -t. -k2 -n | tail -1)
    [ -n "$CUDA_MOD" ] || { echo "[ollama-cuda][FATAL] no cuda/12.x module on $(hostname)"; exit 1; }
    module load "$CUDA_MOD" || { echo "[ollama-cuda][FATAL] module load $CUDA_MOD failed"; exit 1; }
    echo "[ollama-cuda] loaded $CUDA_MOD (CUDA_PATH=$CUDA_PATH)"
fi

SYS_LIB="${CUDA_PATH:-/usr/local/software/$CUDA_MOD}/lib64"
[ -d "$SYS_LIB" ] || SYS_LIB="${CUDA_PATH:-}/targets/x86_64-linux/lib"
BUNDLED_DIR="$OLLAMA_DIR/lib/ollama/cuda_v12"
[ -d "$BUNDLED_DIR" ] || { echo "[ollama-cuda][FATAL] bundled cuda dir missing: $BUNDLED_DIR"; exit 1; }
# Snapshot the inherited LD path ONCE so repeated ollama_apply_cuda calls don't pile up.
: "${_OLLAMA_LD_BASE:=$LD_LIBRARY_PATH}"

# Apply one CUDA lib config by re-pointing the bundled cuda_v12 .so.12 symlinks and
# rebuilding LD_LIBRARY_PATH. Reads the knob env vars at CALL time. Returns 0/!=0.
ollama_apply_cuda() {
    local triplet_src
    if [ "${OLLAMA_BUNDLED:-0}" = 1 ]; then
        ln -sfn libcublas.so.12.8.5.5 "$BUNDLED_DIR/libcublas.so.12"
        ln -sfn libcublasLt.so.12.8.5.5.bundled_bak "$BUNDLED_DIR/libcublasLt.so.12"
        ln -sfn libcudart.so.12.8.90 "$BUNDLED_DIR/libcudart.so.12"
        triplet_src="$BUNDLED_DIR"
        echo "[ollama-cuda] triplet source: BUNDLED 12.8 (no swap)"
    else
        if [ "${OLLAMA_FORCE_SYSTEM:-0}" = 1 ]; then
            triplet_src="$SYS_LIB"; echo "[ollama-cuda] triplet source: SYSTEM $CUDA_MOD ($SYS_LIB)"
        elif [ -e "${OLLAMA_VENDOR_LIB:-$OLLAMA_DIR/lib/ollama/vendor_cuda/lib123}/libcublasLt.so.12" ]; then
            triplet_src="${OLLAMA_VENDOR_LIB:-$OLLAMA_DIR/lib/ollama/vendor_cuda/lib123}"
            echo "[ollama-cuda] triplet source: VENDORED ($triplet_src)"
        else
            triplet_src="$SYS_LIB"; echo "[ollama-cuda][warn] vendored libs missing, using system $CUDA_MOD ($SYS_LIB)"
        fi
        local lib src_so
        for lib in libcublas.so.12 libcublasLt.so.12 libcudart.so.12; do
            src_so=$(readlink -f "$triplet_src/$lib" 2>/dev/null)
            [ -e "$src_so" ] || { echo "[ollama-cuda][FATAL] $lib not found under $triplet_src"; return 1; }
            ln -sfn "$src_so" "$BUNDLED_DIR/$lib" || { echo "[ollama-cuda][FATAL] symlink $lib failed"; return 1; }
            echo "[ollama-cuda] $lib -> $(readlink -f "$BUNDLED_DIR/$lib")"
        done
    fi
    rm -f "$BUNDLED_DIR/libcublasLt.so.12.symlink_bak" 2>/dev/null || true
    export OLLAMA_BIN
    # 2026-06-11: do NOT append $_OLLAMA_LD_BASE. The job loads `module load python/3.8`
    # before sourcing this, which prepends intel-MKL-2020 + intel-compiler libs to the
    # inherited LD. Those get pulled in as transitive deps of cuBLAS and SIGILL on EVERY
    # ampere node (the "node lottery" was actually this). ollama needs only its own libs +
    # the chosen cuda triplet; the system driver (libcuda) resolves from the default ld
    # cache. Keep LD minimal & clean. (run_judges.py imports no MKL, so nothing else needs it.)
    export LD_LIBRARY_PATH="$triplet_src:$BUNDLED_DIR:$OLLAMA_DIR/lib/ollama:$SYS_LIB"
    export OLLAMA_MODELS=/home/$USER/rds/hpc-work/ollama_models
    OLLAMA_TRIPLET_SRC="$triplet_src"
    return 0
}

# Apply the selected (winner/default) config once now, for back-compat with callers
# that just `source` this file and start serve themselves.
ollama_apply_cuda || { echo "[ollama-cuda][FATAL] initial config apply failed"; exit 1; }
echo "[ollama-cuda] ready: OLLAMA_BIN=$OLLAMA_BIN"
