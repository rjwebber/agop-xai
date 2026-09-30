#!/usr/bin/env python3
"""Run controlled AGOP initial-state interventions in the fresh ZC model.

The fresh event checkpoint is deliberately staggered: its SST anomaly is one
model step older than the coarse thermocline/current fields used by the CNN.
This driver therefore patches a *copy* of the release-certified Fortran and
applies a one-time intervention inside ``SSTA``, after the SST update and just
before the compact CNN input is written.  The requested four-field observation
is exact at that instant.

Coarse H1/U1/V1 changes are also lifted to HB/UB/V using the minimum-Euclidean-
norm right inverse of the public ZAVG averaging operator.  This makes the
perturbation persist into the native ocean arrays, but it is still an explicitly
regularized observation-time impulse rather than a uniquely balanced ZC state.
The untouched and zero-dose branches must reproduce all 13 released fields
bit-for-bit before any nonzero branch is accepted.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))

import benchmark_fresh_xai_methods as fresh_xai_benchmark  # noqa: E402
import generate_fresh_zc_dataset as fresh_generator  # noqa: E402

from zc_xai.data import FRESH_SCHEMA_VERSION, ZCData  # noqa: E402
from zc_xai.io import (  # noqa: E402
    load_json,
    sha256_array,
    sha256_file,
    write_json,
    write_npz,
)
from zc_xai.training import ExperimentSpec, load_experiment  # noqa: E402
from zc_xai.xai import AgopExplainer  # noqa: E402

LOGGER = logging.getLogger(__name__)
SCRIPT_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1

N_COARSE_LATITUDES = 20
N_COARSE_LONGITUDES = 27
NATIVE_NYP = 116
NATIVE_NXP = 79
ALLOCATED_NYP = 117
ALLOCATED_NXP = 85
CORE_FIELD_INDICES = np.asarray((0, 5, 6, 7), dtype=np.int64)
CORE_FIELD_NAMES = (
    "sst_anomaly",
    "thermocline_depth",
    "zonal_ocean_current",
    "meridional_ocean_current",
)

CALL_ANCHOR = """c//////////////////////////////////////////////////////
*Eli: writing SST into a file for working with GRADS software,
"""
CALL_REPLACEMENT = """c//////////////////////////////////////////////////////
c One-time controlled observation/native-ocean intervention.  The routine
c is inert unless agop_intervention.bin is present in the run directory.
      CALL APPLY_AGOP_INTERVENTION(T)

*Eli: writing SST into a file for working with GRADS software,
"""

FORTRAN_INTERVENTION = r"""

c=======================================================================
c One-time AGOP intervention hook added by run_zc_agop_interventions.py.
c This source lives only in an isolated experiment workspace.
c=======================================================================
      SUBROUTINE APPLY_AGOP_INTERVENTION(T)
      INCLUDE 'zeq.common'
      INCLUDE 'modified_means.common'

      COMMON/ZDATA/WM(30,34,2,12),DIVM(30,34,12),Q0O(30,34),
     A SSTM(30,34,12),UO(30,34),VO(30,34),DO(30,34),HTAU(34,30,2),
     B WEM(30,34,12),TO(30,34),UV(30,34,2,12),US(30,34),VS(30,34),
     C WP(30,34),DT1(30,34,7),TT(30,34),UV1(30,34),UV2(30,34),
     D WM1(30,34),UAT(30,34),VAT(30,34),DIVT(30,34)
      COMMON/ZDAT2/H1(30,34),U1(30,34),V1(30,34)
      COMMON/commTIME/IT,MP,TY

      REAL T
      REAL TARGET_TO(30,34),TARGET_H1(30,34)
      REAL TARGET_U1(30,34),TARGET_V1(30,34)
      REAL DELTA_HB(117,85),DELTA_UB(117,85),DELTA_V(117,85)
      LOGICAL FILE_EXISTS,ALREADY_CHECKED
      INTEGER IOS,I,J
      SAVE ALREADY_CHECKED
      DATA ALREADY_CHECKED/.FALSE./

      IF(ALREADY_CHECKED) RETURN
      ALREADY_CHECKED=.TRUE.
      INQUIRE(FILE='agop_intervention.bin',EXIST=FILE_EXISTS)
      IF(.NOT.FILE_EXISTS) RETURN
      IF(NSTART.NE.3) STOP 'AGOP intervention requires NSTART=3'

      OPEN(UNIT=97,FILE='agop_intervention.bin',FORM='UNFORMATTED',
     A ACCESS='STREAM',STATUS='OLD',ACTION='READ',IOSTAT=IOS)
      IF(IOS.NE.0) STOP 'Could not open AGOP intervention payload'
      READ(97,IOSTAT=IOS) TARGET_TO,TARGET_H1,TARGET_U1,TARGET_V1,
     A DELTA_HB,DELTA_UB,DELTA_V
      CLOSE(97)
      IF(IOS.NE.0) STOP 'Could not read complete AGOP payload'

c Assign the exact requested CNN observation on the active coarse grid.
      DO 10 J=6,32
      DO 11 I=6,25
         TO(I,J)=TARGET_TO(I,J)
         H1(I,J)=TARGET_H1(I,J)
         U1(I,J)=TARGET_U1(I,J)
         V1(I,J)=TARGET_V1(I,J)
 11   CONTINUE
 10   CONTINUE

c Apply the documented minimum-norm ZAVG lift to the native ocean arrays.
      DO 20 J=1,NXP
      DO 21 I=1,NYP
         HB(I,J)=HB(I,J)+DELTA_HB(I,J)
         UB(I,J)=UB(I,J)+DELTA_UB(I,J)
         V(I,J)=V(I,J)+DELTA_V(I,J)
 21   CONTINUE
 20   CONTINUE

c Restore the native total-SST diagnostic and its 30 C cap after changing TO.
      DO 30 J=6,32
      DO 31 I=6,25
         TT(I,J)=TO(I,J)+SSTM_limit(I,J,IT)
     A        +TY*(SSTM_limit(I,J,MP)-SSTM_limit(I,J,IT))
         IF(TT(I,J).GT.SST_limit) THEN
            TO(I,J)=TO(I,J)+SST_limit-TT(I,J)
            TT(I,J)=SST_limit
         ENDIF
 31   CONTINUE
 30   CONTINUE

      WRITE(6,100) T
 100  FORMAT(' APPLIED ONE-TIME AGOP INTERVENTION AT T=',F15.6)
      RETURN
      END
"""


@dataclass(frozen=True)
class NativeBase:
    label: str
    input_index: int
    checkpoint: Path


@dataclass(frozen=True)
class InterventionCase:
    label: str
    family: str
    base: NativeBase
    coefficient: float | None
    target_standardized: np.ndarray | None


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def nonnegative_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and nonnegative")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/zc-v3"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/zc-v3"))
    parser.add_argument(
        "--agop-benchmark-dir",
        type=Path,
        default=Path(
            "outputs/fresh_agop_benchmark/"
            "core4-cnn-lead-10m-seed-000042-refs-all-batch-1024"
        ),
    )
    parser.add_argument(
        "--generation-workspace",
        type=Path,
        default=Path("outputs/zc_generation/zc-v3"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/zc_agop_interventions/core4-cnn-lead-10m-seed-000042"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lead-months", type=positive_int, default=10)
    parser.add_argument(
        "--subtract-coefficients",
        nargs="+",
        type=nonnegative_float,
        default=(0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--add-coefficients",
        nargs="+",
        type=nonnegative_float,
        default=(0.25, 0.5, 1.0),
    )
    parser.add_argument(
        "--jobs", type=positive_int, default=min(4, os.cpu_count() or 1)
    )
    parser.add_argument(
        "--skip-data-checksums",
        action="store_true",
        help="Development only; this is recorded in the report.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _replace_one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one {label} anchor, found {count}")
    return text.replace(old, new, 1)


def _copy_and_compile_source(
    certified_source: Path,
    output_dir: Path,
    *,
    jobs: int,
) -> tuple[Path, dict[str, Any]]:
    source = output_dir / "source"
    if source.exists():
        shutil.rmtree(source)
    shutil.copytree(
        certified_source,
        source,
        ignore=shutil.ignore_patterns("*.o", "zeqfc1", ".DS_Store"),
    )
    ssta_path = source / "ssta.F"
    original = ssta_path.read_text()
    patched = _replace_one(original, CALL_ANCHOR, CALL_REPLACEMENT, "SSTA hook")
    patched += FORTRAN_INTERVENTION
    ssta_path.write_text(patched)

    environment = fresh_generator.compiler_info()
    flags = "-std=legacy -O3 -I. -ffixed-line-length-none"
    link_flags = (
        f"-isysroot {environment['sdk_path']}" if environment.get("sdk_path") else ""
    )
    command = [
        environment["make"],
        f"-j{jobs}",
        f"FORTRAN={environment['compiler']}",
        f"FLAGS={flags}",
        f"LINKFLAGS={link_flags}",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=source, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    (output_dir / "build.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Intervention build failed; see {output_dir / 'build.log'}")
    executable = source / "zeqfc1"
    return executable, {
        "command": [str(item) for item in command],
        "compiler": environment,
        "flags": flags,
        "link_flags": link_flags,
        "elapsed_seconds": elapsed,
        "patched_ssta_sha256": sha256_file(ssta_path),
        "executable_sha256": sha256_file(executable),
    }


def zavg_operator() -> scipy.sparse.csr_matrix:
    """Return the exact active-grid ZAVG map for HB/UB/V perturbations."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for coarse_row in range(N_COARSE_LATITUDES):
        for coarse_column in range(N_COARSE_LONGITUDES):
            fortran_i = 25 - coarse_row
            fortran_j = 6 + coarse_column
            center_i = 120 - 4 * fortran_i
            x_value = float(fortran_j - 1) * 0.5625
            center_j = int((10.0 * x_value + 0.25 - 20.0) / 2.0)
            row = coarse_row * N_COARSE_LONGITUDES + coarse_column
            for native_i in range(center_i - 2, center_i + 3):
                for native_j in range(center_j - 1, center_j + 2):
                    if not (
                        1 <= native_i <= NATIVE_NYP and 1 <= native_j <= NATIVE_NXP
                    ):
                        raise RuntimeError(
                            "ZAVG stencil escaped the serialized ocean grid"
                        )
                    column = (native_i - 1) + NATIVE_NYP * (native_j - 1)
                    rows.append(row)
                    columns.append(column)
                    values.append(1.0 / 15.0)
    matrix = scipy.sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(
            N_COARSE_LATITUDES * N_COARSE_LONGITUDES,
            NATIVE_NYP * NATIVE_NXP,
        ),
        dtype=np.float64,
    )
    if np.any(np.diff(matrix.indptr) != 15):
        raise RuntimeError("Every ZAVG row must contain exactly 15 averaging weights")
    return matrix


def minimum_norm_lift(
    matrix: scipy.sparse.csr_matrix,
    gram_factor: tuple[np.ndarray, bool],
    coarse_delta: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(coarse_delta, dtype=np.float64)
    if values.shape != (N_COARSE_LATITUDES, N_COARSE_LONGITUDES):
        raise ValueError("A coarse ZAVG delta must have shape (20, 27)")
    weights = scipy.linalg.cho_solve(gram_factor, values.reshape(-1))
    compact = np.asarray(matrix.T @ weights, dtype=np.float64)
    residual = np.asarray(matrix @ compact).reshape(values.shape) - values
    full = np.zeros((ALLOCATED_NYP, ALLOCATED_NXP), dtype=np.float32, order="F")
    full[:NATIVE_NYP, :NATIVE_NXP] = compact.reshape(
        NATIVE_NYP, NATIVE_NXP, order="F"
    ).astype(np.float32)
    realized_float32 = np.asarray(
        matrix @ full[:NATIVE_NYP, :NATIVE_NXP].reshape(-1, order="F"),
        dtype=np.float64,
    ).reshape(values.shape)
    return full, {
        "maximum_absolute_residual_float64": float(np.max(np.abs(residual))),
        "maximum_absolute_residual_after_float32_payload": float(
            np.max(np.abs(realized_float32 - values))
        ),
        "native_l2_norm": float(np.linalg.norm(compact)),
        "native_rms": float(np.sqrt(np.mean(compact**2))),
    }


def _coarse_fortran_array(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if source.shape != (N_COARSE_LATITUDES, N_COARSE_LONGITUDES):
        raise ValueError("A physical field target must have shape (20, 27)")
    output = np.zeros((30, 34), dtype=np.float32, order="F")
    for row in range(N_COARSE_LATITUDES):
        output[25 - row - 1, 5:32] = source[row]
    return output


def _write_payload(
    path: Path,
    *,
    physical_target: np.ndarray,
    physical_base: np.ndarray,
    matrix: scipy.sparse.csr_matrix,
    gram_factor: tuple[np.ndarray, bool],
) -> dict[str, Any]:
    target = np.asarray(physical_target, dtype=np.float32)
    base = np.asarray(physical_base, dtype=np.float32)
    if target.shape != (4, 20, 27) or base.shape != target.shape:
        raise ValueError("Core physical states must have shape (4, 20, 27)")
    native_arrays = []
    lift_reports = {}
    for name, field in zip(CORE_FIELD_NAMES[1:], range(1, 4), strict=True):
        native, diagnostic = minimum_norm_lift(
            matrix,
            gram_factor,
            target[field].astype(np.float64) - base[field].astype(np.float64),
        )
        native_arrays.append(native)
        lift_reports[name] = diagnostic
    arrays = [*(_coarse_fortran_array(field) for field in target), *native_arrays]
    with path.open("wb") as stream:
        for array in arrays:
            stream.write(np.asarray(array, dtype="<f4").tobytes(order="F"))
        stream.flush()
        os.fsync(stream.fileno())
    expected_bytes = (4 * 30 * 34 + 3 * ALLOCATED_NYP * ALLOCATED_NXP) * 4
    if path.stat().st_size != expected_bytes:
        raise RuntimeError("The intervention payload has an unexpected size")
    return {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "zavg_lifts": lift_reports,
    }


def _processed_fields(data: ZCData, start: int, count: int) -> np.ndarray:
    indices = np.arange(start, start + count, dtype=np.int64)
    return np.stack(
        [np.asarray(field[indices], dtype=np.float32) for field in data.arrays],
        axis=1,
    )


def _nino3(data: ZCData, core_fields: np.ndarray) -> np.ndarray:
    latitude = (data.latitudes >= -5.0) & (data.latitudes <= 5.0)
    longitude = (data.longitudes >= 210.0) & (data.longitudes <= 270.0)
    sst = np.asarray(core_fields[:, 0], dtype=np.float64)
    return sst[:, latitude][:, :, longitude].mean(axis=(1, 2))


def _run_directory(
    source: Path,
    executable: Path,
    run_dir: Path,
    *,
    checkpoint: Path,
    input_index: int,
    lead_steps: int,
    payload_source: Path | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    spinup_steps = 100 * 36
    input_nt = spinup_steps + 1 + input_index
    pre_input_nt = input_nt - 1
    pre_time = fresh_generator.model_time(pre_input_nt)
    target_time = fresh_generator.model_time(input_nt + lead_steps)
    fresh_generator.prepare_run(
        source,
        executable,
        run_dir,
        nstart=3,
        tfind=pre_time,
        tzero=pre_time,
        tend=target_time,
        ntape=0,
        nrewnd=11,
        nic=0,
        write_start=pre_time + 0.1,
        write_end=target_time,
        restart=checkpoint,
    )
    if payload_source is not None:
        shutil.copy2(payload_source, run_dir / "agop_intervention.bin")
    runtime = fresh_generator.run_model(run_dir)
    fields = np.asarray(fresh_generator.fresh_memmap(run_dir / "fresh_fields.data"))
    if fields.shape[0] != lead_steps + 1:
        raise RuntimeError(
            f"Expected {lead_steps + 1} states, received {fields.shape[0]}"
        )
    return fields.copy(), runtime


def _extract_authentic_checkpoint(
    *,
    label: str,
    input_index: int,
    generation_workspace: Path,
    destination: Path,
) -> dict[str, Any]:
    report_path = generation_workspace / "production_report.json"
    report = load_json(report_path)
    production_history = generation_workspace / "runs" / "production" / "outhst"
    certified_source = generation_workspace / "source"
    executable = certified_source / "zeqfc1"
    if sha256_file(production_history) != report.get("history_sha256"):
        raise ValueError("Production history failed its recorded SHA-256 check")
    chunk_bytes = int(report["checkpoint_chunk_bytes"])
    checkpoint_steps = int(report["configuration"]["checkpoint_steps"])
    pre_input_nt = 100 * 36 + input_index
    sparse_nt = (pre_input_nt // checkpoint_steps) * checkpoint_steps
    sparse_index = sparse_nt // checkpoint_steps - 1
    checkpoint_dir = destination.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sparse = checkpoint_dir / f"{label}_sparse.hst"
    fresh_generator.extract_checkpoint(
        production_history,
        sparse,
        sparse_index,
        chunk_bytes,
    )
    replay_dir = checkpoint_dir.parent / "checkpoint_replay" / label
    sparse_time = fresh_generator.model_time(sparse_nt)
    exact_time = fresh_generator.model_time(pre_input_nt)
    fresh_generator.prepare_run(
        certified_source,
        executable,
        replay_dir,
        nstart=3,
        tfind=sparse_time,
        tzero=sparse_time,
        tend=exact_time,
        ntape=0,
        nrewnd=11,
        nic=0,
        write_start=exact_time + 1.0,
        write_end=exact_time + 1.0,
        restart=sparse,
    )
    runtime = fresh_generator.run_model(replay_dir)
    shutil.copy2(replay_dir / "outhst", destination)
    return {
        "input_index": input_index,
        "pre_input_nt": pre_input_nt,
        "sparse_nt": sparse_nt,
        "sparse_chunk_index_zero_based": sparse_index,
        "checkpoint_file": destination.name,
        "checkpoint_sha256": sha256_file(destination),
        "checkpoint_size_bytes": destination.stat().st_size,
        "source_history_sha256": report["history_sha256"],
        "certified_executable_sha256": sha256_file(executable),
        "replay_runtime": runtime,
    }


def _phase_matched_neutral_index(
    data: ZCData,
    standardizer: Any,
    event_index: int,
) -> tuple[int, dict[str, Any]]:
    train_start, train_stop = data.metadata["chronological_split"]["train"]
    event_phase = np.asarray(data.phase[event_index], dtype=np.float32)
    candidates = np.arange(train_start, train_stop, dtype=np.int64)
    phase = np.asarray(data.phase[candidates])
    same_phase = np.all(np.isclose(phase, event_phase[None], atol=2.0e-6), axis=1)
    candidates = candidates[same_phase]
    if candidates.size == 0:
        raise RuntimeError("No training state shares the event annual phase")
    values = data.load_inputs(candidates, standardizer=standardizer)[:, :-2]
    rms = np.sqrt(np.mean(values.astype(np.float64) ** 2, axis=1))
    position = int(np.argmin(rms))
    selected = int(candidates[position])
    return selected, {
        "selection": (
            "authentic training state at the exact event phase nearest the "
            "coordinatewise training spatial mean in standardized RMS distance"
        ),
        "candidate_count": int(candidates.size),
        "candidate_indices_sha256": sha256_array(candidates),
        "event_phase_raw": event_phase.astype(float).tolist(),
        "selected_spatial_standardized_rms": float(rms[position]),
    }


def _model_forecast(model: torch.nn.Module, standardized: np.ndarray) -> float:
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(standardized, dtype=np.float32)[None])
        return float(model(tensor).detach().cpu().item())


def _make_figure(
    output: Path,
    trajectories: dict[str, np.ndarray],
    case_reports: dict[str, dict[str, Any]],
    *,
    steps_per_month: int,
) -> None:
    months = np.arange(next(iter(trajectories.values())).size) / steps_per_month
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6), sharey=True)
    extreme = axes[0]
    average = axes[1]
    extreme.plot(
        months,
        trajectories["extreme_disabled"],
        color="black",
        lw=2.4,
        label="Unmodified",
    )
    subtract_labels = sorted(
        (name for name in trajectories if name.startswith("extreme_minus_")),
        key=lambda name: case_reports[name]["coefficient"],
    )
    add_labels = sorted(
        (name for name in trajectories if name.startswith("extreme_plus_")),
        key=lambda name: case_reports[name]["coefficient"],
    )
    for index, name in enumerate(subtract_labels):
        coefficient = case_reports[name]["coefficient"]
        extreme.plot(
            months,
            trajectories[name],
            color=plt.cm.Blues(0.45 + 0.5 * (index + 1) / len(subtract_labels)),
            lw=1.45,
            label=rf"Subtract ${coefficient:g}\,q\hat e$",
        )
    for index, name in enumerate(add_labels):
        coefficient = case_reports[name]["coefficient"]
        extreme.plot(
            months,
            trajectories[name],
            color=plt.cm.Reds(0.45 + 0.5 * (index + 1) / len(add_labels)),
            lw=1.45,
            label=rf"Add ${coefficient:g}\,q\hat e$",
        )
    extreme.axhline(0.0, color="0.75", lw=0.8)
    extreme.set_title("Extreme El Ni\N{LATIN SMALL LETTER N WITH TILDE}o restart")
    extreme.set_xlabel("Months after intervention")
    extreme.set_ylabel("Ni\N{LATIN SMALL LETTER N WITH TILDE}o-3 anomaly ($^\circ$C)")
    extreme.legend(loc="best", fontsize=7.7, ncol=2)

    average.plot(
        months,
        trajectories["neutral_disabled"],
        color="0.35",
        lw=2.0,
        label="Authentic phase-matched neutral state",
    )
    average.plot(
        months,
        trajectories["neutral_mean_plus_projection"],
        color="#6a3d9a",
        lw=2.2,
        label=r"Training mean $+q\hat e$",
    )
    average.plot(
        months,
        trajectories["neutral_state_plus_projection"],
        color="#1b9e77",
        lw=1.8,
        label=r"Authentic neutral state $+q\hat e$",
    )
    average.axhline(0.0, color="0.75", lw=0.8)
    average.set_title("Average-state controls")
    average.set_xlabel("Months after intervention")
    average.legend(loc="best", fontsize=8.0)
    for axis in axes:
        axis.grid(color="0.9", lw=0.6)
        axis.set_xlim(months[0], months[-1])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.lead_months != 10:
        raise ValueError("The certified event checkpoint currently supports 10 months")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    started_utc = datetime.now(UTC).isoformat()

    data = ZCData(
        args.data_dir,
        input_profile="core4",
        verify_checksums=not args.skip_data_checksums,
    )
    if data.metadata.get("schema_version") != FRESH_SCHEMA_VERSION:
        raise ValueError("This intervention requires the fresh zc-v3 data set")
    all_fields_data = ZCData(
        args.data_dir, input_profile="all13", verify_checksums=False
    )
    spec = ExperimentSpec(
        architecture="cnn",
        lead_months=args.lead_months,
        train_years=10_000.0,
        seed=args.seed,
        input_profile="core4",
    )
    experiment = load_experiment(data, args.artifacts_dir, spec, device="cpu")
    fixed = data.fixed_supervised_split(args.lead_months)
    factor, factor_provenance = fresh_xai_benchmark.load_validated_full_agop_factor(
        args.agop_benchmark_dir,
        data=data,
        experiment=experiment,
        fixed_training_inputs=fixed.train_inputs,
    )

    event_metadata = next(
        entry
        for entry in data.metadata["event_restart_checkpoints"]
        if entry["label"] == "extreme_el_nino"
    )
    event_index = int(event_metadata["input_index"])
    target_index = int(event_metadata["target_index"])
    event_checkpoint = data.data_dir / event_metadata["checkpoint_file"]
    if sha256_file(event_checkpoint) != event_metadata["checkpoint_sha256"]:
        raise ValueError("The extreme-event checkpoint failed its SHA-256 check")

    event_standardized = data.load_inputs(
        np.asarray([event_index]), standardizer=experiment.standardizer
    )[0].astype(np.float64)
    explanation = AgopExplainer(factor).explain(event_standardized[None])[0]
    spatial_explanation = explanation[:-2]
    spatial_norm = float(np.linalg.norm(spatial_explanation))
    feasible_direction = spatial_explanation / spatial_norm
    projection = float(event_standardized[:-2] @ feasible_direction)
    if projection <= 0.0:
        raise RuntimeError("The PSD AGOP projection was unexpectedly nonpositive")

    neutral_index, neutral_selection = _phase_matched_neutral_index(
        data,
        experiment.standardizer,
        event_index,
    )
    checkpoints_dir = output_dir / "checkpoints"
    neutral_checkpoint = checkpoints_dir / "phase_matched_neutral_pre_input.hst"
    neutral_checkpoint_report = _extract_authentic_checkpoint(
        label="phase_matched_neutral",
        input_index=neutral_index,
        generation_workspace=args.generation_workspace.expanduser().resolve(),
        destination=neutral_checkpoint,
    )
    event_base = NativeBase("extreme", event_index, event_checkpoint)
    neutral_base = NativeBase("neutral", neutral_index, neutral_checkpoint)

    certified_source = args.generation_workspace.expanduser().resolve() / "source"
    certified_executable = certified_source / "zeqfc1"
    if (
        sha256_file(certified_executable)
        != event_metadata["event_replay_executable_sha256"]
    ):
        raise ValueError("The certified executable does not match event provenance")
    executable, build_report = _copy_and_compile_source(
        certified_source,
        output_dir,
        jobs=args.jobs,
    )
    patched_source = output_dir / "source"

    matrix = zavg_operator()
    gram = np.asarray((matrix @ matrix.T).toarray(), dtype=np.float64)
    gram_factor = scipy.linalg.cho_factor(gram, lower=True, check_finite=True)
    operator_report = {
        "shape": list(matrix.shape),
        "nonzero_count": int(matrix.nnz),
        "row_rank": int(np.linalg.matrix_rank(gram)),
        "gram_condition_number_2": float(np.linalg.cond(gram)),
        "inverse": "B.T @ solve(B @ B.T, delta), minimum Euclidean norm",
    }

    neutral_standardized = data.load_inputs(
        np.asarray([neutral_index]), standardizer=experiment.standardizer
    )[0].astype(np.float64)
    cases: list[InterventionCase] = [
        InterventionCase(
            "extreme_disabled", "disabled_control", event_base, None, None
        ),
        InterventionCase(
            "extreme_zero",
            "zero_dose_control",
            event_base,
            0.0,
            event_standardized.copy(),
        ),
        InterventionCase(
            "neutral_disabled", "disabled_control", neutral_base, None, None
        ),
        InterventionCase(
            "neutral_zero",
            "zero_dose_control",
            neutral_base,
            0.0,
            neutral_standardized.copy(),
        ),
    ]
    for coefficient in sorted(set(args.subtract_coefficients)):
        target = event_standardized.copy()
        target[:-2] -= coefficient * projection * feasible_direction
        cases.append(
            InterventionCase(
                f"extreme_minus_{coefficient:g}".replace(".", "p"),
                "extreme_subtraction",
                event_base,
                coefficient,
                target,
            )
        )
    for coefficient in sorted(set(args.add_coefficients)):
        target = event_standardized.copy()
        target[:-2] += coefficient * projection * feasible_direction
        cases.append(
            InterventionCase(
                f"extreme_plus_{coefficient:g}".replace(".", "p"),
                "extreme_addition",
                event_base,
                coefficient,
                target,
            )
        )
    exact_mean_target = event_standardized.copy()
    exact_mean_target[:-2] = projection * feasible_direction
    cases.append(
        InterventionCase(
            "neutral_mean_plus_projection",
            "coordinatewise_training_mean_plus_projection",
            neutral_base,
            1.0,
            exact_mean_target,
        )
    )
    neutral_plus_target = neutral_standardized.copy()
    neutral_plus_target[:-2] += projection * feasible_direction
    cases.append(
        InterventionCase(
            "neutral_state_plus_projection",
            "authentic_neutral_state_plus_projection",
            neutral_base,
            1.0,
            neutral_plus_target,
        )
    )

    payloads_dir = output_dir / "payloads"
    runs_dir = output_dir / "runs"
    payloads_dir.mkdir()
    trajectories: dict[str, np.ndarray] = {}
    core_trajectories: dict[str, np.ndarray] = {}
    case_reports: dict[str, dict[str, Any]] = {}
    baseline_fields: dict[str, np.ndarray] = {}
    lead_steps = args.lead_months * data.steps_per_month

    for case in cases:
        LOGGER.info("Running %s", case.label)
        physical_base = np.stack(
            [np.asarray(field[case.base.input_index]) for field in data.arrays], axis=0
        ).astype(np.float32)
        payload_path = None
        payload_report = None
        physical_target = physical_base
        if case.target_standardized is not None:
            if case.family == "zero_dose_control":
                # Do not rely on a standardize/inverse-standardize round trip
                # for the bitwise c=0 safety gate.
                physical_target = physical_base.copy()
            else:
                physical_target = (
                    experiment.standardizer.inverse_transform(
                        np.asarray(case.target_standardized, dtype=np.float32)
                    )[:-2]
                    .reshape(4, 20, 27)
                    .astype(np.float32)
                )
            payload_path = payloads_dir / f"{case.label}.bin"
            payload_report = _write_payload(
                payload_path,
                physical_target=physical_target,
                physical_base=physical_base,
                matrix=matrix,
                gram_factor=gram_factor,
            )
        fields, runtime = _run_directory(
            patched_source,
            executable,
            runs_dir / case.label,
            checkpoint=case.base.checkpoint,
            input_index=case.base.input_index,
            lead_steps=lead_steps,
            payload_source=payload_path,
        )
        if not np.isfinite(fields).all():
            raise RuntimeError(f"Nonfinite model output in intervention {case.label}")
        core = fields[:, CORE_FIELD_INDICES]
        nino3 = _nino3(data, core)
        trajectories[case.label] = nino3
        core_trajectories[case.label] = core
        expected = _processed_fields(
            all_fields_data, case.base.input_index, lead_steps + 1
        )
        exact_replay = bool(np.array_equal(fields, expected))
        if (
            case.family in {"disabled_control", "zero_dose_control"}
            and not exact_replay
        ):
            maximum = float(
                np.max(np.abs(fields.astype(np.float64) - expected.astype(np.float64)))
            )
            raise RuntimeError(
                f"Safety gate failed for {case.label}: maximum replay error {maximum}"
            )
        if case.family == "disabled_control":
            baseline_fields[case.base.label] = fields
        initial_residual = core[0].astype(np.float64) - physical_target.astype(
            np.float64
        )
        initial_standardized_residual = (
            initial_residual / experiment.standardizer.scale[:-2].reshape(4, 20, 27)
        )
        forecast = (
            _model_forecast(experiment.model, case.target_standardized)
            if case.target_standardized is not None
            else _model_forecast(
                experiment.model,
                data.load_inputs(
                    np.asarray([case.base.input_index]),
                    standardizer=experiment.standardizer,
                )[0],
            )
        )
        report = {
            "label": case.label,
            "family": case.family,
            "base": case.base.label,
            "base_input_index": case.base.input_index,
            "coefficient": case.coefficient,
            "payload": payload_report,
            "cnn_initial_forecast_nino3_c": forecast,
            "zc_initial_nino3_c": float(nino3[0]),
            "zc_final_nino3_c": float(nino3[-1]),
            "unmodified_true_target_nino3_c": float(
                data.target[case.base.input_index + lead_steps]
            ),
            "initial_target_maximum_absolute_error_physical": float(
                np.max(np.abs(initial_residual))
            ),
            "initial_target_rms_error_standardized": float(
                np.sqrt(np.mean(initial_standardized_residual**2))
            ),
            "initial_target_phase_raw": np.asarray(
                data.phase[case.base.input_index], dtype=float
            ).tolist(),
            "atmosphere_small_nino3_reset_condition_at_initial_state": bool(
                abs(nino3[0]) <= 0.1
            ),
            "all_13_fields_finite": bool(np.isfinite(fields).all()),
            "initial_total_sst_points_at_30c_cap": int(
                np.count_nonzero(fields[0, 9] >= np.float32(30.0))
            ),
            "exact_all13_replay_of_unmodified_release": exact_replay,
            "runtime": runtime,
            "fresh_fields_sha256": sha256_file(
                runs_dir / case.label / "fresh_fields.data"
            ),
            "final_restart_sha256": sha256_file(runs_dir / case.label / "outhst"),
        }
        case_reports[case.label] = report

    for label, core in core_trajectories.items():
        if label in {
            "extreme_disabled",
            "extreme_zero",
            "neutral_disabled",
            "neutral_zero",
        }:
            continue
        base_label = case_reports[label]["base"]
        baseline = baseline_fields[base_label][:, CORE_FIELD_INDICES]
        delta = core.astype(np.float64) - baseline.astype(np.float64)
        standardized_delta = delta / experiment.standardizer.scale[:-2].reshape(
            1, 4, 20, 27
        )
        flat_delta = standardized_delta.reshape(standardized_delta.shape[0], -1)
        standardized_norm = np.linalg.norm(flat_delta, axis=1)
        alignment = np.divide(
            flat_delta @ feasible_direction,
            standardized_norm,
            out=np.zeros_like(standardized_norm),
            where=standardized_norm > 0.0,
        )
        case_reports[label]["realized_core_delta_rms_by_step"] = np.sqrt(
            np.mean(delta**2, axis=(1, 2, 3))
        ).tolist()
        case_reports[label]["realized_standardized_delta"] = {
            "l2_norm_by_step": standardized_norm.tolist(),
            "rms_by_step": (
                standardized_norm / math.sqrt(feasible_direction.size)
            ).tolist(),
            "cosine_with_positive_agop_direction_by_step": alignment.tolist(),
            "field_rms_by_step": np.sqrt(
                np.mean(standardized_delta**2, axis=(2, 3))
            ).tolist(),
        }

    arrays: dict[str, Any] = {
        "event_input_index": np.asarray(event_index, dtype=np.int64),
        "event_target_index": np.asarray(target_index, dtype=np.int64),
        "neutral_input_index": np.asarray(neutral_index, dtype=np.int64),
        "projection_q": np.asarray(projection, dtype=np.float64),
        "feasible_spatial_direction": feasible_direction.astype(np.float64),
    }
    for label, values in trajectories.items():
        arrays[f"nino3__{label}"] = values.astype(np.float64)
        arrays[f"core4__{label}"] = core_trajectories[label].astype(np.float32)
    trajectory_path = output_dir / "trajectories.npz"
    write_npz(trajectory_path, overwrite=True, **arrays)
    figure_path = output_dir / "nino3_intervention_trajectories.pdf"
    _make_figure(
        figure_path,
        trajectories,
        case_reports,
        steps_per_month=data.steps_per_month,
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "status": "complete",
        "started_utc": started_utc,
        "completed_utc": datetime.now(UTC).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "scientific_definition": {
            "intervention_time": (
                "inside SSTA after the input-time SST update/cap and immediately "
                "before the compact CNN observation is written"
            ),
            "phase_policy": "hold the two event clock/phase coordinates fixed",
            "direction": (
                "unit AGOP vector at the extreme test input, projected onto and "
                "renormalized within the 2160 spatial core4 coordinates"
            ),
            "projection_q": projection,
            "projection_formula": (
                "q = dot(z_extreme_spatial, e_spatial / ||e_spatial||)"
            ),
            "mean_plus_formula": "z_spatial = q * e_spatial / ||e_spatial||",
            "extreme_formula": "z_spatial = z_extreme_spatial +/- c*q*direction",
            "native_lift": operator_report,
            "qualification": (
                "The ZAVG inverse is nonunique. This experiment uses the explicit "
                "minimum-Euclidean-norm lift and is an observation-time impulse, "
                "not a uniquely balanced native ZC initial condition. Native "
                "UBNDY/HBNDY boundary-memory arrays are held fixed; the model is "
                "allowed to rebalance the lifted state dynamically."
            ),
        },
        "event": event_metadata,
        "neutral_state": {
            "input_index": neutral_index,
            **neutral_selection,
            "checkpoint": neutral_checkpoint_report,
        },
        "agop": {
            **factor_provenance,
            "full_unit_direction_phase_squared_mass": float(
                np.sum(explanation[-2:] ** 2)
            ),
            "spatial_component_norm": spatial_norm,
            "spatial_projection_q": projection,
            "spatial_kick_rms_at_c1": float(projection / math.sqrt(2160.0)),
        },
        "data": data.provenance(),
        "processed_checksums_verified": not args.skip_data_checksums,
        "model": {
            "artifact_directory": str(experiment.artifact_dir),
            "checkpoint_sha256": experiment.checkpoint_sha256,
            "spec": {
                "architecture": "cnn",
                "input_profile": "core4",
                "lead_months": args.lead_months,
                "seed": args.seed,
            },
        },
        "source": {
            "certified_directory": str(certified_source),
            "certified_executable_sha256": sha256_file(certified_executable),
            "patched_build": build_report,
            "platform": platform.platform(),
        },
        "safety_gates": {
            "extreme_disabled_all13_bitwise": case_reports["extreme_disabled"][
                "exact_all13_replay_of_unmodified_release"
            ],
            "extreme_zero_all13_bitwise": case_reports["extreme_zero"][
                "exact_all13_replay_of_unmodified_release"
            ],
            "neutral_disabled_all13_bitwise": case_reports["neutral_disabled"][
                "exact_all13_replay_of_unmodified_release"
            ],
            "neutral_zero_all13_bitwise": case_reports["neutral_zero"][
                "exact_all13_replay_of_unmodified_release"
            ],
        },
        "cases": case_reports,
        "outputs": {
            "trajectories_file": trajectory_path.name,
            "trajectories_sha256": sha256_file(trajectory_path),
            "figure_pdf": figure_path.name,
            "figure_pdf_sha256": sha256_file(figure_path),
            "figure_png": figure_path.with_suffix(".png").name,
            "figure_png_sha256": sha256_file(figure_path.with_suffix(".png")),
        },
    }
    write_json(output_dir / "report.json", report, overwrite=True)
    LOGGER.info("Complete: %s", output_dir / "report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
