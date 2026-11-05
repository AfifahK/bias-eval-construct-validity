#!/bin/bash
# Sourced AFTER setup_ollama_cuda.sh. Provides ollama_serve_resilient(): start
# `ollama serve`, verify it survives AND drives the GPU (gemma3:1b < SPEED_LIMIT s);
# if the current config SIGILLs or falls back to CPU, cycle the other known configs
# until one works ON THIS NODE. Leaves a healthy serve running and exports
# OLLAMA_PID. Returns 1 only if every config fails on this node.
#
# Caters to the node-specific flakiness: the proven winner (system swap) is tried
# first, then vendored 12.3, bundled 12.8, bundled-no-module.

ollama_serve_resilient() {
    local speed_limit="${OLLAMA_SPEED_LIMIT:-30}"
    local probe_model="${OLLAMA_PROBE_MODEL:-gemma3:1b}"
    local host="${OLLAMA_HOST:-127.0.0.1:11434}"
    local jid="${SLURM_JOB_ID:-$$}${SLURM_ARRAY_TASK_ID:+_${SLURM_ARRAY_TASK_ID}}"
    # name | knob-assignments (applied via env then ollama_apply_cuda)
    local configs=(
        "vendored124|OLLAMA_FORCE_SYSTEM=0 OLLAMA_BUNDLED=0 OLLAMA_VENDOR_LIB=$OLLAMA_DIR/lib/ollama/vendor_cuda/lib124"
        "system121|OLLAMA_FORCE_SYSTEM=1 OLLAMA_BUNDLED=0"
        "vendored123|OLLAMA_FORCE_SYSTEM=0 OLLAMA_BUNDLED=0 OLLAMA_VENDOR_LIB=$OLLAMA_DIR/lib/ollama/vendor_cuda/lib123"
        "bundled128|OLLAMA_BUNDLED=1"
        "bundled_nomod|OLLAMA_BUNDLED=1 OLLAMA_EXTRA_MODULE=__none__"
    )
    local entry name kv
    for entry in "${configs[@]}"; do
        name="${entry%%|*}"; kv="${entry#*|}"
        echo "[serve-resilient] === trying config: $name ($kv) ==="
        # reset knobs, then apply this config's knobs
        unset OLLAMA_FORCE_SYSTEM OLLAMA_BUNDLED OLLAMA_VENDOR_LIB OLLAMA_EXTRA_MODULE
        export $kv
        ollama_apply_cuda || { echo "[serve-resilient:$name] apply failed, next"; continue; }
        local slog="logs/ollama_serve_${jid}_${name}.log"
        "$OLLAMA_BIN" serve > "$slog" 2>&1 &
        OLLAMA_PID=$!
        sleep 15
        if ! kill -0 $OLLAMA_PID 2>/dev/null; then
            echo "[serve-resilient:$name] serve DIED <15s (SIGILL?). tail:"; tail -8 "$slog" | sed 's/^/    /'
            continue
        fi
        local t0 resp dur a
        # Probe with retries: first call may include model load. Try up to 4x.
        # Use curl with a SANITIZED lib path (env -u LD_LIBRARY_PATH) because the
        # cuda/python modules inject OpenSSL 1.1.1 that breaks system curl's
        # libk5crypto (EVP_KDF_ctrl). If curl is still unusable, fall back to a
        # python urllib probe (python requests works in this env). ollama serve
        # keeps the cuda LD path; only the probe client is sanitized.
        resp=""; dur=999
        for a in 1 2 3 4; do
            t0=$SECONDS
            resp=$(env -u LD_LIBRARY_PATH curl -s --max-time 120 "http://${host}/api/generate" \
                -d "{\"model\":\"$probe_model\",\"prompt\":\"Reply one word: ok\",\"stream\":false}" 2>/dev/null)
            echo "$resp" | grep -q '"response"' && { dur=$((SECONDS - t0)); break; }
            # curl unusable or empty -> try python urllib (no LD_LIBRARY_PATH issues)
            if [ -z "$resp" ]; then
                resp=$(env -u LD_LIBRARY_PATH python3 - "$host" "$probe_model" 2>/dev/null <<'PYEOF'
import sys,json,urllib.request
host,model=sys.argv[1],sys.argv[2]
try:
    req=urllib.request.Request("http://%s/api/generate"%host,
        data=json.dumps({"model":model,"prompt":"Reply one word: ok","stream":False}).encode(),
        headers={"Content-Type":"application/json"})
    print(urllib.request.urlopen(req,timeout=120).read().decode())
except Exception as e:
    print("")
PYEOF
)
                echo "$resp" | grep -q '"response"' && { dur=$((SECONDS - t0)); break; }
            fi
            echo "[serve-resilient:$name] probe attempt $a no response, retrying..."
            sleep 5
        done
        if ! echo "$resp" | grep -q '"response"'; then
            echo "[serve-resilient:$name] no response after retries, killing. resp=${resp:0:120}"
            kill $OLLAMA_PID 2>/dev/null; wait $OLLAMA_PID 2>/dev/null; continue
        fi
        if [ "$dur" -ge "$speed_limit" ]; then
            echo "[serve-resilient:$name] ${dur}s >= ${speed_limit}s = CPU fallback, killing"
            kill $OLLAMA_PID 2>/dev/null; wait $OLLAMA_PID 2>/dev/null; continue
        fi
        echo "[serve-resilient:$name] *** HEALTHY: ${dur}s GPU on $(hostname), PID=$OLLAMA_PID ***"
        export OLLAMA_PID OLLAMA_WORKING_CONFIG="$name"
        return 0
    done
    echo "[serve-resilient][FATAL] no CUDA config drove the GPU on $(hostname)"
    return 1
}
