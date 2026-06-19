#!/usr/bin/env python3
"""Distribución de LongYAAL (CU/NCA y CA) por SEGMENTO y por TOKEN para los
estudios de políticas de emisión, agregando los tres decoders (rambr, prunembr,
rerank) dentro de cada policy.

Script unificado para ASR y MT: selecciona la tarea con ``--task {asr,mt,both}``
(por defecto ``both``). Vive en ``src/cascade_2026/`` y localiza los datos de
``experiments/policies_trades/<task>/`` de forma relativa a su propia ubicación,
de modo que el estudio es reproducible con independencia del directorio de
trabajo: al ejecutarlo sin argumentos escanea las carpetas ``omnisteval_<task>_*``
de todas las policies y vuelca, para cada tarea, el detalle por segmento y por
token en su carpeta ``latencies_<task>_study``.
"""

import argparse
import glob
import json
import os
import re
from statistics import mean, median, pstdev


# --------------------------------------------------------------------------- #
#  Rutas (relativas a la ubicación del script -> reproducible)                 #
# --------------------------------------------------------------------------- #
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # src/cascade_2026
REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
POLICIES_DIR = os.path.join(REPO_ROOT, "experiments", "policies_trades")

# Configuración por tarea: prefijo del run_tag a eliminar y carpeta de salida.
TASKS = {
    "asr": {
        "glob": os.path.join(POLICIES_DIR, "asr", "*", "results", "omnisteval_asr_*"),
        "out_dir": os.path.join(POLICIES_DIR, "asr", "latencies_asr_study"),
        "prefix": "asr_",
    },
    "mt": {
        "glob": os.path.join(POLICIES_DIR, "mt", "*", "results", "omnisteval_mt_*"),
        "out_dir": os.path.join(POLICIES_DIR, "mt", "latencies_mt_study"),
        "prefix": "mt_",
    },
}

DECODERS = ("prunembr_xcomet", "rambr_chrf", "rerank_kiwi")


def yaal_one(delays, source_length, reference_length, recording_end, is_longform=True):
    if not delays or source_length is None or source_length <= 0:
        return None
    rec_end = float("inf") if recording_end is None else recording_end
    if delays[0] >= rec_end or (not is_longform and delays[0] >= source_length):
        return None
    tgt_len = reference_length if reference_length else len(delays)
    gamma = max(len(delays), tgt_len) / source_length
    yaal = 0.0
    tau = 0
    for t_minus_1, d in enumerate(delays):
        if d >= rec_end or (not is_longform and d >= source_length):
            break
        yaal += d - t_minus_1 / gamma
        tau = t_minus_1 + 1
    return yaal / tau if tau > 0 else None


def token_lags(delays, source_length, reference_length, recording_end, is_longform=True):
    out = []
    if not delays or source_length is None or source_length <= 0:
        return out
    rec_end = float("inf") if recording_end is None else recording_end
    if delays[0] >= rec_end or (not is_longform and delays[0] >= source_length):
        return out
    tgt_len = reference_length if reference_length else len(delays)
    gamma = max(len(delays), tgt_len) / source_length
    for i, d in enumerate(delays):
        if d >= rec_end or (not is_longform and d >= source_length):
            break
        out.append((i, d - i / gamma))
    return out


def read_run(instances_path):
    """Devuelve el detalle por segmento del run:
       - seg_cu/seg_ca: YAAL por segmento (variante CU/CA)
       - tok_cu/tok_ca: lag por token
       - segments: lista con (seg_idx, variant, yaal) por segmento válido
       - tokens: lista con (seg_idx, variant, tok_idx, lag) por token
       - conteo de segmentos totales y descartados por variante."""
    seg_cu, seg_ca, tok_cu, tok_ca = [], [], [], []
    segments, tokens = [], []
    n_seg = 0
    disc_cu, disc_ca = 0, 0
    with open(instances_path, encoding="utf-8") as f:
        for seg_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ref = d.get("reference") or ""
            ref_len = len(ref.split(" "))
            src_len = d.get("source_length")
            rec_end = d.get("time_to_recording_end")
            n_seg += 1

            cu = yaal_one(d.get("emission_cu"), src_len, ref_len, rec_end)
            ca = yaal_one(d.get("emission_ca"), src_len, ref_len, rec_end)
            if cu is not None:
                seg_cu.append(cu)
                segments.append((seg_idx, "cu", cu))
            else:
                disc_cu += 1
            if ca is not None:
                seg_ca.append(ca)
                segments.append((seg_idx, "ca", ca))
            else:
                disc_ca += 1

            for tok_idx, lag in token_lags(d.get("emission_cu"), src_len, ref_len, rec_end):
                tok_cu.append(lag)
                tokens.append((seg_idx, "cu", tok_idx, lag))
            for tok_idx, lag in token_lags(d.get("emission_ca"), src_len, ref_len, rec_end):
                tok_ca.append(lag)
                tokens.append((seg_idx, "ca", tok_idx, lag))

    return {"seg_cu": seg_cu, "seg_ca": seg_ca,
            "tok_cu": tok_cu, "tok_ca": tok_ca,
            "segments": segments, "tokens": tokens,
            "n_seg": n_seg, "disc_cu": disc_cu, "disc_ca": disc_ca}


def _stats(vals):
    keys = ("n", "mean", "median", "std", "min", "p10", "p50", "p90", "p95", "p99", "max")
    if not vals:
        return {k: float("nan") for k in keys}
    s = sorted(vals)
    pct = lambda p: s[min(len(s) - 1, max(0, int(round(p / 100 * (len(s) - 1)))))]
    return {
        "n": len(vals), "mean": mean(vals), "median": median(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals), "p10": pct(10), "p50": pct(50),
        "p90": pct(90), "p95": pct(95), "p99": pct(99), "max": max(vals),
    }


def policy_of(run_tag, prefix):
    """Extrae la policy quitando el prefijo de tarea y el sufijo del decoder + nN_segM.
       asr_tolerant_tau1_prunembr_xcomet_n8_seg960 -> tolerant_tau1
       mt_mt_waitk3_rambr_chrf_n8_seg960           -> mt_waitk3
       mt_mt_hybrid_k3_tau1_rerank_kiwi_n8_seg960  -> mt_hybrid_k3_tau1"""
    tag = run_tag
    if tag.startswith(prefix):
        tag = tag[len(prefix):]
    for dec in DECODERS:
        tag = re.sub(rf"_{dec}_n\d+_seg\d+$", "", tag)
    return tag


def _run_tag(instances_path):
    d = os.path.basename(os.path.dirname(instances_path))
    return d.replace("omnisteval_", "")


def process(paths, prefix):
    """Acumula valores crudos por policy (juntando los tres decoders) y devuelve
    ``(agg, seg_rows, tok_rows)``."""
    agg = {}  # policy -> dict de listas + conteos
    seg_rows = []  # (policy, run_tag, seg_idx, variant, yaal)
    tok_rows = []  # (policy, run_tag, seg_idx, variant, tok_idx, lag)
    for p in paths:
        run_tag = _run_tag(p)
        pol = policy_of(run_tag, prefix)
        r = read_run(p)
        a = agg.setdefault(pol, {"seg_cu": [], "seg_ca": [], "tok_cu": [], "tok_ca": [],
                                 "n_seg": 0, "disc_cu": 0, "disc_ca": 0})
        for k in ("seg_cu", "seg_ca", "tok_cu", "tok_ca"):
            a[k] += r[k]
        for k in ("n_seg", "disc_cu", "disc_ca"):
            a[k] += r[k]
        for seg_idx, variant, yaal in r["segments"]:
            seg_rows.append((pol, run_tag, seg_idx, variant, yaal))
        for seg_idx, variant, tok_idx, lag in r["tokens"]:
            tok_rows.append((pol, run_tag, seg_idx, variant, tok_idx, lag))
    return agg, seg_rows, tok_rows


def report(agg):
    """Imprime el resumen por policy en stdout."""
    for pol in sorted(agg):
        a = agg[pol]
        frac_cu = a["disc_cu"] / a["n_seg"] if a["n_seg"] else float("nan")
        frac_ca = a["disc_ca"] / a["n_seg"] if a["n_seg"] else float("nan")
        print(f"\n=== {pol} ===  (segmentos totales={a['n_seg']}, "
              f"descartados CU={a['disc_cu']} [{frac_cu:.1%}], CA={a['disc_ca']} [{frac_ca:.1%}])")
        for level in ("seg", "tok"):
            for variant in ("cu", "ca"):
                s = _stats(a[f"{level}_{variant}"])
                lvl_name = "segmento" if level == "seg" else "token   "
                print(f"  [{lvl_name}] {variant.upper()}  n={s['n']:>6}  "
                      f"min={s['min']:8.1f}  p10={s['p10']:8.1f}  p50={s['p50']:8.1f}  "
                      f"mean={s['mean']:8.1f}  p90={s['p90']:8.1f}  p95={s['p95']:8.1f}  "
                      f"p99={s['p99']:8.1f}  max={s['max']:8.1f}")


def write_segment_csv(path, seg_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("policy,run_tag,segment_idx,variant,yaal\n")
        for pol, run_tag, seg_idx, variant, yaal in seg_rows:
            f.write(f"{pol},{run_tag},{seg_idx},{variant},{yaal:.4f}\n")
    print(f"[ok] detalle por segmento ({len(seg_rows)} filas) -> {path}")


def write_token_csv(path, tok_rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("policy,run_tag,segment_idx,variant,token_idx,lag\n")
        for pol, run_tag, seg_idx, variant, tok_idx, lag in tok_rows:
            f.write(f"{pol},{run_tag},{seg_idx},{variant},{tok_idx},{lag:.4f}\n")
    print(f"[ok] detalle por token ({len(tok_rows)} filas) -> {path}")


def _collect_paths(pattern):
    return sorted(
        os.path.join(d, "instances.resegmented.jsonl")
        for d in glob.glob(pattern)
        if os.path.isfile(os.path.join(d, "instances.resegmented.jsonl"))
    )


def run_task(task, segment_csv=None, token_csv=None):
    """Procesa una tarea (asr/mt) con sus rutas por defecto y vuelca los CSV de
    detalle por segmento y por token en su carpeta ``latencies_<task>_study``."""
    cfg = TASKS[task]
    paths = _collect_paths(cfg["glob"])
    if not paths:
        raise SystemExit(f"[{task}] No se encontró ningún instances.resegmented.jsonl "
                         f"con el patrón {cfg['glob']}")
    print(f"\n########## TAREA: {task.upper()} ##########")
    print(f"[info] {len(paths)} runs encontrados -> {cfg['glob']}")
    agg, seg_rows, tok_rows = process(paths, cfg["prefix"])
    report(agg)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    seg_path = segment_csv or os.path.join(out_dir, f"latency_per_segment_{task}.csv")
    tok_path = token_csv or os.path.join(out_dir, f"latency_per_token_{task}.csv")
    write_segment_csv(seg_path, seg_rows)
    write_token_csv(tok_path, tok_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["asr", "mt", "both"], default="both",
                    help="tarea a procesar (por defecto: both)")
    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument("--instances", help="un instances.resegmented.jsonl concreto")
    g.add_argument("--glob", help="patrón de carpetas omnisteval_* (omite rutas por defecto)")
    ap.add_argument("--prefix", default=None,
                    help="prefijo de tarea a quitar del run_tag (p.ej. asr_ o mt_); "
                         "requerido junto a --instances/--glob")
    ap.add_argument("--segment-csv", default=None, help="ruta del CSV por segmento")
    ap.add_argument("--token-csv", default=None, help="ruta del CSV por token")
    args = ap.parse_args()

    # Modo manual: un instances o glob explícito.
    if args.instances or args.glob:
        if not args.prefix:
            ap.error("--prefix es obligatorio con --instances/--glob")
        paths = [args.instances] if args.instances else _collect_paths(args.glob)
        if not paths:
            raise SystemExit("No se encontró ningún instances.resegmented.jsonl")
        agg, seg_rows, tok_rows = process(paths, args.prefix)
        report(agg)
        if args.segment_csv:
            write_segment_csv(args.segment_csv, seg_rows)
        if args.token_csv:
            write_token_csv(args.token_csv, tok_rows)
        return

    # Modo por defecto (botón Play): tarea(s) con rutas reproducibles.
    tasks = ["asr", "mt"] if args.task == "both" else [args.task]
    for task in tasks:
        run_task(task, segment_csv=args.segment_csv, token_csv=args.token_csv)


if __name__ == "__main__":
    main()
