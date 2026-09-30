#!/usr/bin/env python3
"""Build and run a reproducible, interpretation-first Zebiak--Cane data set.

The upstream Fortran source is copied into an isolated workspace and patched
there; this repository does not redistribute the upstream source.  The patch
adds a compact 13-field output on the active 20 x 27 grid and extends native
history records with the complete evolving state and global clock needed for
bitwise continuation under the certified Standard configuration. It does not
alter the model equations.

The default production experiment discards 100 years and retains 12,000
years at the native three-steps-per-month cadence.  Sparse complete native
checkpoints are written every ten years.  Once the run is complete, exact
pre-input checkpoints are selected for the strongest retained El Nino and La
Nina targets at a ten-month lead and certified by replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_VERSION = "1.1.0"
GENERATION_SCHEMA_VERSION = "fresh-zc-generation-v2"
DATA_SCHEMA_VERSION = "fresh-zc-interpretability-v1"
UPSTREAM_URL = (
    "https://groups.seas.harvard.edu/climate/eli/Downloads/CZ_model_share.zip"
)
UPSTREAM_ZIP_SHA256 = "4b737051d7c9f27c54531ef565c19a99c7edf47a07fffdb9fbc976327bf318ee"
REFERENCE_CONTINUOUS_STREAM_SHA256 = (
    "46ca95ead9db93ef023d35ea4a447f105926d74a5a9431ac92655b3e7d16bb9f"
)
REFERENCE_TOOLCHAIN = {
    "compiler_version": "GNU Fortran (Homebrew GCC 13.1.0) 13.1.0",
    "platform": "macOS-26.6.2-arm64-arm-64bit",
    "machine": "arm64",
    "byteorder": "little",
}

DTYPE = np.dtype("<f4")
N_LATITUDE = 20
N_LONGITUDE = 27
VALUES_PER_FIELD = N_LATITUDE * N_LONGITUDE
FIELD_RECORD_BYTES = VALUES_PER_FIELD * DTYPE.itemsize
EFFECTIVE_DTD_MONTHS = float(np.float32(1.0 / 3.0))
TZERO_MONTHS = 0.5
STEPS_PER_MONTH = 3
STEPS_PER_YEAR = 36
FIRST_FROM_REST_OUTPUT_NT = 2

LATITUDES = np.arange(-19.0, 19.1, 2.0, dtype=np.float64)
LONGITUDES = 129.375 + 5.625 * np.arange(N_LONGITUDE, dtype=np.float64)
NINO3_LATITUDE_MASK = (LATITUDES >= -5.0) & (LATITUDES <= 5.0)
NINO3_LONGITUDE_MASK = (LONGITUDES >= 210.0) & (LONGITUDES <= 270.0)

FIELDS: tuple[dict[str, Any], ...] = (
    {
        "name": "sst_anomaly",
        "fortran": "TO",
        "units": "degrees C",
        "group": "core",
        "role": "native prognostic SST anomaly",
    },
    {
        "name": "zonal_wind_stress",
        "fortran": "HTAU(:,:,1)",
        "units": "model wind-stress units",
        "group": "legacy",
        "role": "diagnosed atmospheric wind-stress anomaly",
    },
    {
        "name": "meridional_wind_stress",
        "fortran": "HTAU(:,:,2)",
        "units": "model wind-stress units",
        "group": "legacy",
        "role": "diagnosed atmospheric wind-stress anomaly",
    },
    {
        "name": "zonal_surface_wind",
        "fortran": "UO",
        "units": "model atmosphere-wind units",
        "group": "legacy",
        "role": "diagnosed steady-atmosphere zonal wind anomaly",
    },
    {
        "name": "meridional_surface_wind",
        "fortran": "VO",
        "units": "model atmosphere-wind units",
        "group": "legacy",
        "role": "diagnosed steady-atmosphere meridional wind anomaly",
    },
    {
        "name": "thermocline_depth",
        "fortran": "H1",
        "units": "m",
        "group": "core",
        "role": "coarse diagnostic of native ocean thermocline-depth anomaly",
    },
    {
        "name": "zonal_ocean_current",
        "fortran": "U1",
        "units": "model depth-averaged ocean-current units",
        "group": "core",
        "role": "coarse diagnostic of native depth-averaged ocean current",
    },
    {
        "name": "meridional_ocean_current",
        "fortran": "V1",
        "units": "model depth-averaged ocean-current units",
        "group": "core",
        "role": "coarse diagnostic of native depth-averaged ocean current",
    },
    {
        "name": "atmospheric_heating",
        "fortran": "QF",
        "units": "nondimensional model heating units",
        "group": "legacy",
        "role": "diagnosed atmospheric heating anomaly",
    },
    {
        "name": "total_sst",
        "fortran": "TT",
        "units": "degrees C",
        "group": "legacy",
        "role": "SST anomaly plus seasonal background, subject to the model cap",
    },
    {
        "name": "zonal_mixed_layer_current",
        "fortran": "US",
        "units": "cm s-1",
        "group": "process",
        "role": "diagnosed mixed-layer zonal current anomaly used by SST physics",
    },
    {
        "name": "meridional_mixed_layer_current",
        "fortran": "VS",
        "units": "cm s-1",
        "group": "process",
        "role": "diagnosed mixed-layer meridional current anomaly used by SST physics",
    },
    {
        "name": "upwelling_anomaly",
        "fortran": "WP",
        "units": "model upwelling units",
        "group": "process",
        "role": "diagnosed upwelling anomaly used by SST physics",
    },
)

CORE_FIELDS = tuple(field["name"] for field in FIELDS if field["group"] == "core")
LEGACY_TEN_FIELDS = tuple(field["name"] for field in FIELDS[:10])
PROCESS_FIELDS = tuple(field["name"] for field in FIELDS if field["group"] == "process")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    result.add_argument("mode", choices=("preflight", "generate", "finalize"))
    result.add_argument("--source-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--workspace", type=Path, required=True)
    result.add_argument("--spinup-years", type=int, default=100)
    result.add_argument("--retained-years", type=int, default=12_000)
    result.add_argument("--checkpoint-years", type=int, default=10)
    result.add_argument("--event-lead-months", type=int, default=10)
    result.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    result.add_argument("--overwrite", action="store_true")
    result.add_argument(
        "--keep-native-field-stream",
        action="store_true",
        help="Keep the temporary interleaved fresh_fields.data after conversion.",
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = parser().parse_args(argv)
    for name in ("spinup_years", "retained_years", "checkpoint_years", "jobs"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.event_lead_months <= 0:
        raise SystemExit("--event-lead-months must be positive")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def generator_identity() -> dict[str, str]:
    script = Path(__file__).resolve()
    return {
        "script_file": script.name,
        "script_version": SCRIPT_VERSION,
        "generation_schema_version": GENERATION_SCHEMA_VERSION,
        "script_sha256": sha256_file(script),
    }


def manifest_sha256(manifest: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} differs from the certified value")


def replace_one(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} patch anchor, found {count}")
    return text.replace(old, new)


def replace_line(text: str, name: str, value: str) -> str:
    updated, count = re.subn(
        rf"^{re.escape(name)}\s*=.*$", f"{name:<7}=  {value}", text, flags=re.MULTILINE
    )
    if count != 1:
        raise RuntimeError(f"Expected exactly one {name} line, found {count}")
    return updated


def source_manifest(source_dir: Path) -> dict[str, str]:
    ignored = {".DS_Store", "zeqfc1"}
    manifest: dict[str, str] = {}
    for path in sorted(source_dir.rglob("*")):
        if path.is_file() and path.name not in ignored and path.suffix != ".o":
            manifest[str(path.relative_to(source_dir))] = sha256_file(path)
    return manifest


def patch_source(build_dir: Path) -> dict[str, str]:
    """Add compact output and complete fresh-run restart state."""

    main_path = build_dir / "zeq1main.F"
    main_source = main_path.read_text()
    main_source = replace_one(
        main_source,
        "      IF(NSTART.GT.0) CALL RDHIST(0,TFIND)\n",
        "      IF(NSTART.GT.0) CALL RDHIST(0,TFIND)\n"
        "      IF(NSTART.EQ.3) THEN\n"
        "        TD_restart=TD\n"
        "        NT_restart=NT\n"
        "        TZERO_restart=TZERO\n"
        "      ENDIF\n",
        "restart time capture",
    )
    main_source = replace_one(
        main_source,
        "c if nstart=1 or nstart=2, time is set to tzero in input data file......\n"
        "c otherwise (nstart=3) time is gotten from history file.................\n"
        "      IF(NSTART.EQ.1.OR.NSTART.EQ.2) GO TO 100\n"
        "      TD=TZERO\n"
        "100   CONTINUE\n"
        "      NT=0\n",
        "c Preserve original global time arithmetic on a native restart.\n"
        "c This is required for bitwise seasonal interpolation at long times.\n"
        "      IF(NSTART.EQ.3) THEN\n"
        "        TD=TD_restart\n"
        "        NT=NT_restart\n"
        "        TZERO=TZERO_restart\n"
        "      ELSE\n"
        "        IF(NSTART.EQ.0) TD=TZERO\n"
        "        NT=0\n"
        "      ENDIF\n"
        "100   CONTINUE\n",
        "restart time restoration",
    )
    main_path.write_text(main_source)

    openfl_path = build_dir / "openfl.F"
    openfl = openfl_path.read_text()
    openfl = replace_one(
        openfl,
        "      CHARACTER*60 FN61,FN62,FN63\n",
        "      CHARACTER*60 FN61,FN62,FN63\n      common/fresh_output/ISTEP_fresh\n",
        "fresh output declaration",
    )
    openfl = replace_one(
        openfl,
        "      ISTEP_grads=1\n",
        "      ISTEP_grads=1\n      ISTEP_fresh=1\n",
        "fresh output initialization",
    )
    openfl = replace_one(
        openfl,
        "     &     )\n\n      RETURN\n",
        "     &     )\n\n"
        "c Interpretation-first output: one 20x27 direct-access record per field.\n"
        "      OPEN(UNIT=49,FILE='fresh_fields.data',FORM='UNFORMATTED',\n"
        "     $     ACCESS='DIRECT',RECL=20*27*4,STATUS='REPLACE')\n\n"
        "      RETURN\n",
        "fresh output file",
    )
    openfl_path.write_text(openfl)

    ssta_path = build_dir / "ssta.F"
    ssta = ssta_path.read_text()
    ssta = replace_one(
        ssta,
        "      common/IST/ist\n",
        "      common/IST/ist\n      common/fresh_output/ISTEP_fresh\n",
        "fresh output common",
    )
    ssta = replace_one(
        ssta,
        "      logical froze_background\n",
        "      logical froze_background,write_legacy_grads\n",
        "legacy output switch declaration",
    )
    write_lines = (
        ("TO", "((TO(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("HTAU(:,:,1)", "((HTAU(JJ,31-II,1),JJ=6,32),II=25,6,-1)"),
        ("HTAU(:,:,2)", "((HTAU(JJ,31-II,2),JJ=6,32),II=25,6,-1)"),
        ("UO", "((UO(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("VO", "((VO(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("H1", "((H1(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("U1", "((U1(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("V1", "((V1(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("QF", "((QF(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("TT", "((TT(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("US", "((US(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("VS", "((VS(II,JJ),JJ=6,32),II=25,6,-1)"),
        ("WP", "((WP(II,JJ),JJ=6,32),II=25,6,-1)"),
    )
    fresh_block = [
        "c Compact active-domain fields; ordering is documented by the Python driver."
    ]
    for label, expression in write_lines:
        fresh_block.extend(
            (
                f"c {label}",
                "         WRITE(49,rec=ISTEP_fresh)",
                f"     $        {expression}",
                "         ISTEP_fresh=ISTEP_fresh+1",
            )
        )
    ssta = replace_one(
        ssta,
        "!        is:ie - longitude; js:je - latitude\n",
        "\n".join(fresh_block)
        + "\n\n"
        + "         INQUIRE(FILE='enable_legacy_grads_output',\n"
        + "     $        EXIST=write_legacy_grads)\n"
        + "         if (write_legacy_grads) then\n\n"
        + "!        is:ie - longitude; js:je - latitude\n",
        "fresh field writes",
    )
    ssta = replace_one(
        ssta,
        "         ISTEP_grads= ISTEP_grads + 1\n\n      end if\n\n"
        "c//////////////////////////////////////////////////////\n",
        "         ISTEP_grads= ISTEP_grads + 1\n\n"
        "         end if\n      end if\n\n"
        "c//////////////////////////////////////////////////////\n",
        "legacy output switch close",
    )
    ssta_path.write_text(ssta)

    atmosphere_path = build_dir / "ztmfc1.F"
    atmosphere = atmosphere_path.read_text()
    atmosphere = replace_one(
        atmosphere,
        "      DIMENSION DIF(16,34),Q0(30,34),RDIV(30,34)\n",
        "      DIMENSION DIF(16,34),Q0(30,34),RDIV(30,34)\n",
        "persistent atmosphere convergence state",
    )
    atmosphere = replace_one(
        atmosphere,
        "      COMPLEX D2,U1(80),U(30,64),E1(80),E2(80),DIV(30,64)\n",
        "      COMPLEX D2,U1(80),U(30,64),E1(80),E2(80),DIV(30,64)\n"
        "      COMMON/fresh_zatmc_state/DIF,Q,E1,E2,UBAR\n",
        "persistent atmosphere state common",
    )
    atmosphere = replace_one(
        atmosphere,
        "      COMMON/STRPAR/RSTR,TCUT,TSEED,TAMP1,TAMP2,JLEFT,JRT,ITOP,IBOT,NIC\n"
        "      \n      SAVE\n",
        "      COMMON/STRPAR/RSTR,TCUT,TSEED,TAMP1,TAMP2,JLEFT,JRT,ITOP,IBOT,NIC\n"
        "      COMMON/fresh_stress_state/DSEED,T0,R\n"
        "      \n      SAVE\n",
        "persistent stress state common",
    )
    atmosphere_path.write_text(atmosphere)

    history_path = build_dir / "nrdhist.F"
    history = history_path.read_text()
    history = replace_one(
        history,
        "      INCLUDE 'zeq.common'\n",
        "      INCLUDE 'zeq.common'\n      include 'modified_means.common'\n",
        "extended restart common state",
    )
    history = replace_one(
        history,
        "      COMMON/ZDAT2/H1(30,34),U1(30,34),V1(30,34)\n",
        "      COMMON/ZDAT2/H1(30,34),U1(30,34),V1(30,34)\n"
        "      REAL QF(30,34),QEO(30,64)\n"
        "      COMMON/grads_atm/QF,QEO\n"
        "      REAL DIF(16,34)\n"
        "      COMPLEX Q(80,64),E1(80),E2(80)\n"
        "      COMMON/fresh_zatmc_state/DIF,Q,E1,E2,UBAR\n"
        "      REAL*8 DSEED\n"
        "      COMMON/fresh_stress_state/DSEED,T0,R\n",
        "extended restart declarations",
    )
    history = replace_one(
        history,
        "      READ(INHST,ERR=99,END=100) Q0O,UO,VO,DO,TO,U1,V1,H1\n",
        "      READ(INHST,ERR=99,END=100) Q0O,UO,VO,DO,TO,U1,V1,H1\n"
        "      READ(INHST,ERR=99,END=100) HTAU,US,VS,WP,DT1,TT,\n"
        "     A UV1,UV2,WM1,UAT,VAT,DIVT,QF,QEO,DIF,Q,E1,E2,\n"
        "     A UBAR,DSEED,T0,R,time_of_last_initialization\n",
        "extended restart read",
    )
    history = replace_one(
        history,
        "      WRITE(OUTHST) Q0O,UO,VO,DO,TO,U1,V1,H1\n",
        "      WRITE(OUTHST) Q0O,UO,VO,DO,TO,U1,V1,H1\n"
        "      WRITE(OUTHST) HTAU,US,VS,WP,DT1,TT,UV1,UV2,WM1,\n"
        "     A UAT,VAT,DIVT,QF,QEO,DIF,Q,E1,E2,UBAR,DSEED,T0,R,\n"
        "     A time_of_last_initialization\n",
        "extended restart write",
    )
    history = replace_one(
        history,
        "      READ(INHST,END=100)\n      READ(INHST,END=100)\n"
        "      READ(INHST,END=100)\n1     CONTINUE\n",
        "      READ(INHST,END=100)\n      READ(INHST,END=100)\n"
        "      READ(INHST,END=100)\n      READ(INHST,END=100)\n1     CONTINUE\n",
        "four-record restart skip",
    )
    history_path.write_text(history)

    close_path = build_dir / "close_files.F"
    close_text = close_path.read_text()
    close_text = replace_one(
        close_text,
        "      close(47)\n",
        "      close(47)\n      close(48)\n      close(49)\n",
        "fresh output close",
    )
    close_path.write_text(close_text)
    return {
        name: sha256_file(build_dir / name)
        for name in (
            "zeq1main.F",
            "openfl.F",
            "ssta.F",
            "ztmfc1.F",
            "nrdhist.F",
            "close_files.F",
        )
    }


def compiler_info() -> dict[str, Any]:
    compiler = shutil.which("gfortran")
    make = shutil.which("make")
    if compiler is None or make is None:
        raise RuntimeError("gfortran and make are required")
    version = subprocess.run(
        [compiler, "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    sdk_path = None
    if platform.system() == "Darwin":
        xcrun = shutil.which("xcrun")
        if xcrun is None:
            raise RuntimeError("xcrun is required on macOS")
        sdk_path = subprocess.run(
            [xcrun, "--show-sdk-path"], check=True, capture_output=True, text=True
        ).stdout.strip()
    return {
        "compiler": compiler,
        "compiler_version": version,
        "make": make,
        "sdk_path": sdk_path,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
    }


def prepare_build(
    source_dir: Path, workspace: Path, jobs: int
) -> tuple[Path, dict[str, Any]]:
    build_dir = workspace / "source"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(
        source_dir,
        build_dir,
        ignore=shutil.ignore_patterns("*.o", "zeqfc1", ".DS_Store"),
    )
    original_manifest = source_manifest(source_dir)
    patched = patch_source(build_dir)
    environment = compiler_info()
    flags = "-std=legacy -O3 -I. -ffixed-line-length-none"
    link_flags = (
        f"-isysroot {environment['sdk_path']}" if environment["sdk_path"] else ""
    )
    command = [
        environment["make"],
        f"-j{jobs}",
        f"FORTRAN={environment['compiler']}",
        f"FLAGS={flags}",
        f"LINKFLAGS={link_flags}",
    ]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=build_dir, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "build.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError(f"Compilation failed; see {workspace / 'build.log'}")
    executable = build_dir / "zeqfc1"
    return executable, {
        "upstream": {
            "download_url": UPSTREAM_URL,
            "expected_zip_sha256": UPSTREAM_ZIP_SHA256,
            "local_source_directory": str(source_dir),
            "local_source_manifest_sha256": hashlib.sha256(
                json.dumps(original_manifest, sort_keys=True).encode()
            ).hexdigest(),
            "local_source_files": original_manifest,
            "redistributed_with_processed_data": False,
        },
        "patch": {
            "description": (
                "I/O-only compact fields plus complete fresh-run history extension"
            ),
            "patched_source_sha256": patched,
            "model_equations_changed": False,
        },
        "compiler": environment,
        "command": command,
        "flags": flags,
        "link_flags": link_flags,
        "elapsed_seconds": elapsed,
        "executable_sha256": sha256_file(executable),
    }


def update_fc(
    template: str,
    *,
    nstart: int,
    tfind: float,
    tzero: float,
    tend: float,
    ntape: int,
    nrewnd: int,
    nic: int,
) -> str:
    result = template
    for name, value in (
        ("NSTART", str(nstart)),
        ("TFIND", f"{tfind:.7f}"),
        ("TZERO", f"{tzero:.7f}"),
        ("TENDD", f"{tend:.7f}"),
        ("NTAPE", str(ntape)),
        ("NREWND", str(nrewnd)),
        ("NIC", str(nic)),
    ):
        result = replace_line(result, name, value)
    return result


def update_namelist(template: str, *, write_start: float, write_end: float) -> str:
    replacements = {
        "time_start_writing_grads_data": f"{write_start:.7f}",
        "time_end_writing_grads_data": f"{write_end:.7f}",
        "write_grads_data": ".t.",
    }
    result = template
    for name, value in replacements.items():
        result, count = re.subn(
            rf"^\s*{name}\s*=.*$", f" {name}={value},", result, flags=re.MULTILINE
        )
        if count != 1:
            raise RuntimeError(f"Expected one namelist value for {name}, found {count}")
    return result


def prepare_run(
    source_dir: Path,
    executable: Path,
    run_dir: Path,
    *,
    nstart: int,
    tfind: float,
    tzero: float,
    tend: float,
    ntape: int,
    nrewnd: int,
    nic: int,
    write_start: float,
    write_end: float,
    restart: Path | None = None,
    legacy_grads_output: bool = False,
) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    shutil.copy2(executable, run_dir / "zeqfc1")
    shutil.copytree(source_dir / "Data", run_dir / "Data")
    (run_dir / "EOF_data" / "to").mkdir(parents=True)
    (run_dir / "fc.data").write_text(
        update_fc(
            (source_dir / "fc.data").read_text(),
            nstart=nstart,
            tfind=tfind,
            tzero=tzero,
            tend=tend,
            ntape=ntape,
            nrewnd=nrewnd,
            nic=nic,
        )
    )
    experiment = source_dir / "Experiments" / "Standard"
    (run_dir / "modified_means.namelist").write_text(
        update_namelist(
            (experiment / "modified_means.namelist_1").read_text(),
            write_start=write_start,
            write_end=write_end,
        )
    )
    shutil.copy2(experiment / "scales_EOF.namelist_1", run_dir / "scales_EOF.namelist")
    if restart is not None:
        shutil.copy2(restart, run_dir / "zeq9fsu.hst")
    if legacy_grads_output:
        (run_dir / "enable_legacy_grads_output").touch()


def run_model(run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(
            [str(run_dir / "zeqfc1")], cwd=run_dir, stdout=log, stderr=subprocess.STDOUT
        )
    elapsed = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(f"ZC run failed; see {run_dir / 'run.log'}")
    return {
        "elapsed_seconds": elapsed,
        "fresh_fields_size_bytes": (run_dir / "fresh_fields.data").stat().st_size,
        "history_size_bytes": (run_dir / "outhst").stat().st_size,
        "completed_utc": datetime.now(UTC).isoformat(),
    }


def fresh_memmap(path: Path) -> np.memmap:
    bytes_per_step = len(FIELDS) * FIELD_RECORD_BYTES
    steps, remainder = divmod(path.stat().st_size, bytes_per_step)
    if steps == 0 or remainder:
        raise RuntimeError(f"Invalid compact field stream size: {path}")
    return np.memmap(
        path, dtype=DTYPE, mode="r", shape=(steps, len(FIELDS), N_LATITUDE, N_LONGITUDE)
    )


def legacy_memmap(path: Path) -> np.memmap:
    bytes_per_step = 10 * 30 * 34 * DTYPE.itemsize
    steps, remainder = divmod(path.stat().st_size, bytes_per_step)
    if steps == 0 or remainder:
        raise RuntimeError(f"Invalid legacy field stream size: {path}")
    return np.memmap(path, dtype=DTYPE, mode="r", shape=(steps, 10, 30, 34))


def checkpoint_layout(path: Path, expected_count: int) -> tuple[int, int]:
    size = path.stat().st_size
    chunk, remainder = divmod(size, expected_count)
    if remainder or chunk <= 0:
        raise RuntimeError(
            f"History file size {size} is incompatible with "
            f"{expected_count} checkpoints"
        )
    return chunk, expected_count


def extract_checkpoint(
    history: Path, output: Path, index: int, chunk_bytes: int
) -> None:
    with history.open("rb") as source, output.open("wb") as target:
        source.seek(index * chunk_bytes)
        payload = source.read(chunk_bytes)
        if len(payload) != chunk_bytes:
            raise RuntimeError("Checkpoint extraction reached an unexpected EOF")
        target.write(payload)


def nino3_from_fields(fields: np.ndarray) -> np.ndarray:
    selected = fields[:, 0][:, NINO3_LATITUDE_MASK][:, :, NINO3_LONGITUDE_MASK]
    return np.mean(selected, axis=(1, 2), dtype=np.float64).astype(np.float32)


def model_time(nt: int, tzero: float = TZERO_MONTHS) -> float:
    return float(
        np.float32(
            np.float32(tzero) + np.float32(nt) * np.float32(EFFECTIVE_DTD_MONTHS)
        )
    )


def phase_features(native_time: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * np.mod(native_time - TZERO_MONTHS, 12.0) / 12.0
    return np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    executable, build = prepare_build(args.source_dir.resolve(), workspace, args.jobs)
    run_dir = workspace / "runs" / "preflight_reference"
    duration_years = 150
    tend = TZERO_MONTHS + 12.0 * duration_years
    prepare_run(
        workspace / "source",
        executable,
        run_dir,
        nstart=0,
        tfind=120.5,
        tzero=TZERO_MONTHS,
        tend=tend,
        ntape=STEPS_PER_YEAR,
        nrewnd=10,
        nic=4,
        write_start=1.0,
        write_end=tend,
        legacy_grads_output=True,
    )
    runtime = run_model(run_dir)
    fresh = fresh_memmap(run_dir / "fresh_fields.data")
    legacy = legacy_memmap(run_dir / "grads.data")
    if fresh.shape[0] != legacy.shape[0]:
        raise RuntimeError("Fresh and legacy outputs contain different step counts")
    legacy_active = legacy[:, :, 5:25, 5:32]
    first_ten_equal = bool(np.array_equal(fresh[:, :10], legacy_active))
    if not first_ten_equal:
        by_field = [
            float(np.max(np.abs(fresh[:, i].astype(np.float64) - legacy_active[:, i])))
            for i in range(10)
        ]
        raise RuntimeError(
            f"Compact output does not match legacy active grid: {by_field}"
        )
    if not np.isfinite(fresh).all():
        raise RuntimeError("Compact preflight output contains nonfinite values")

    expected_checkpoints = duration_years
    chunk_bytes, _ = checkpoint_layout(run_dir / "outhst", expected_checkpoints)
    restart_cases = []
    reference_log = (run_dir / "run.log").read_text()
    forced_reset_pattern = re.compile(
        r"initializing ISTEP after 4 years\.; t=\s*([0-9.]+)"
    )
    reference_forced_resets = tuple(
        float(value) for value in forced_reset_pattern.findall(reference_log)
    )
    if not reference_forced_resets:
        raise RuntimeError(
            "Long preflight did not encounter the forced ZATMC initialization guard."
        )
    for case_name, checkpoint_year, continuation_years in (
        ("cross_forced_initialization_guard", 130, 10),
        ("post_forced_initialization_restart", 140, 5),
    ):
        checkpoint = workspace / f"preflight_year{checkpoint_year}.hst"
        extract_checkpoint(
            run_dir / "outhst", checkpoint, checkpoint_year - 1, chunk_bytes
        )
        continuation_dir = workspace / "runs" / f"preflight_{case_name}"
        checkpoint_time = TZERO_MONTHS + 12.0 * checkpoint_year
        continuation_end = checkpoint_time + 12.0 * continuation_years
        prepare_run(
            workspace / "source",
            executable,
            continuation_dir,
            nstart=3,
            tfind=checkpoint_time,
            tzero=checkpoint_time,
            tend=continuation_end,
            ntape=0,
            nrewnd=11,
            nic=0,
            write_start=checkpoint_time + 0.1,
            write_end=continuation_end,
            restart=checkpoint,
        )
        restart_runtime = run_model(continuation_dir)
        restarted = fresh_memmap(continuation_dir / "fresh_fields.data")
        # Reference record zero is NT=2. The first continuation output is
        # checkpoint NT + 1, hence zero-based reference index NT - 1.
        reference_start = checkpoint_year * STEPS_PER_YEAR - 1
        reference = fresh[reference_start : reference_start + restarted.shape[0]]
        replay_equal = bool(np.array_equal(restarted, reference))
        replay_max_error = float(
            np.max(np.abs(restarted.astype(np.float64) - reference.astype(np.float64)))
        )
        if not replay_equal:
            raise RuntimeError(
                f"{case_name} checkpoint failed bitwise continuation: "
                f"max error {replay_max_error}"
            )
        restart_forced_resets = tuple(
            float(value)
            for value in forced_reset_pattern.findall(
                (continuation_dir / "run.log").read_text()
            )
        )
        expected_forced_resets = tuple(
            value
            for value in reference_forced_resets
            if checkpoint_time < value <= continuation_end
        )
        if restart_forced_resets != expected_forced_resets:
            raise RuntimeError(
                f"{case_name} forced-reset timestamps differ: "
                f"{restart_forced_resets!r} versus {expected_forced_resets!r}"
            )
        restart_cases.append(
            {
                "case": case_name,
                "checkpoint_year": checkpoint_year,
                "continuation_years": continuation_years,
                "all_13_fields_bitwise_equal": replay_equal,
                "maximum_absolute_error": replay_max_error,
                "steps_compared": int(restarted.shape[0]),
                "checkpoint_sha256": sha256_file(checkpoint),
                "runtime": restart_runtime,
                "forced_initialization_timestamps_months": list(restart_forced_resets),
                "forced_initialization_timestamps_match_reference": True,
            }
        )

    report = {
        "report_schema_version": GENERATION_SCHEMA_VERSION,
        "generator": generator_identity(),
        "script_version": SCRIPT_VERSION,
        "status": "passed",
        "build": build,
        "requested_production_configuration": production_configuration(args),
        "configuration": {
            "duration_years": duration_years,
            "output_steps": int(fresh.shape[0]),
            "checkpoint_count": expected_checkpoints,
            "checkpoint_chunk_bytes": chunk_bytes,
        },
        "schema": {
            "shape": list(fresh.shape),
            "field_order": [field["name"] for field in FIELDS],
            "first_ten_bitwise_equal_legacy_active_grid": first_ten_equal,
            "all_values_finite": True,
        },
        "restart": {
            "all_cases_all_13_fields_bitwise_equal": all(
                case["all_13_fields_bitwise_equal"] for case in restart_cases
            ),
            "reference_forced_initialization_timestamps_months": list(
                reference_forced_resets
            ),
            "cases": restart_cases,
        },
        "runtime": {"reference": runtime},
    }
    write_json(workspace / "preflight_report.json", report)
    return report


def production_configuration(args: argparse.Namespace) -> dict[str, Any]:
    spinup_steps = args.spinup_years * STEPS_PER_YEAR
    retained_steps = args.retained_years * STEPS_PER_YEAR
    total_steps = spinup_steps + retained_steps
    checkpoint_steps = args.checkpoint_years * STEPS_PER_YEAR
    if total_steps % checkpoint_steps:
        raise RuntimeError(
            "Total duration must be divisible by the checkpoint interval"
        )
    return {
        "spinup_years": args.spinup_years,
        "retained_years": args.retained_years,
        "checkpoint_years": args.checkpoint_years,
        "event_lead_months": args.event_lead_months,
        "spinup_steps": spinup_steps,
        "retained_steps": retained_steps,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_count": total_steps // checkpoint_steps,
        "tend_months": TZERO_MONTHS + total_steps * EFFECTIVE_DTD_MONTHS,
        "write_start_months": TZERO_MONTHS + spinup_steps * EFFECTIVE_DTD_MONTHS + 0.1,
        "steps_per_month": STEPS_PER_MONTH,
        "steps_per_year": STEPS_PER_YEAR,
        "field_count": len(FIELDS),
        "field_order": [field["name"] for field in FIELDS],
        "field_shape": [N_LATITUDE, N_LONGITUDE],
        "field_dtype": DTYPE.str,
        "stream_layout": "time-major, field-major, latitude-major, longitude-major",
        "stream_bytes_per_step": len(FIELDS) * FIELD_RECORD_BYTES,
    }


def validate_preflight_binding(
    args: argparse.Namespace, workspace: Path
) -> tuple[dict[str, Any], Path]:
    report_path = workspace / "preflight_report.json"
    if not report_path.is_file():
        raise RuntimeError(
            "Run a passing preflight with this workspace before generate"
        )
    report = json.loads(report_path.read_text())
    require_equal(report.get("status"), "passed", "preflight status")
    require_equal(
        report.get("report_schema_version"),
        GENERATION_SCHEMA_VERSION,
        "preflight report schema",
    )
    require_equal(report.get("generator"), generator_identity(), "preflight generator")
    require_equal(
        report.get("requested_production_configuration"),
        production_configuration(args),
        "preflight production configuration",
    )
    build = report["build"]
    source_digest = manifest_sha256(source_manifest(args.source_dir.resolve()))
    require_equal(
        source_digest,
        build["upstream"]["local_source_manifest_sha256"],
        "upstream source manifest",
    )
    executable = workspace / "source" / "zeqfc1"
    if not executable.is_file():
        raise RuntimeError(f"Certified executable is missing: {executable}")
    require_equal(
        sha256_file(executable), build["executable_sha256"], "certified executable"
    )
    return report, report_path


def reference_toolchain_matches(build: dict[str, Any]) -> bool:
    compiler = build["compiler"]
    return all(compiler.get(key) == value for key, value in REFERENCE_TOOLCHAIN.items())


def generate(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    preflight, preflight_path = validate_preflight_binding(args, workspace)
    executable = workspace / "source" / "zeqfc1"
    config = production_configuration(args)
    run_dir = workspace / "runs" / "production"
    prepare_run(
        workspace / "source",
        executable,
        run_dir,
        nstart=0,
        tfind=120.5,
        tzero=TZERO_MONTHS,
        tend=config["tend_months"],
        ntape=config["checkpoint_steps"],
        nrewnd=10,
        nic=4,
        write_start=config["write_start_months"],
        write_end=config["tend_months"],
    )
    runtime = run_model(run_dir)
    fields = fresh_memmap(run_dir / "fresh_fields.data")
    if fields.shape[0] != config["retained_steps"]:
        raise RuntimeError(
            f"Expected {config['retained_steps']} retained steps, "
            f"found {fields.shape[0]}"
        )
    stream_sha256 = sha256_file(run_dir / "fresh_fields.data")
    enforce_reference_hash = reference_toolchain_matches(preflight["build"])
    if (
        enforce_reference_hash
        and args.spinup_years == 100
        and args.retained_years == 12_000
        and args.checkpoint_years == 10
        and stream_sha256 != REFERENCE_CONTINUOUS_STREAM_SHA256
    ):
        raise RuntimeError(
            "Restart instrumentation changed the continuous production trajectory: "
            f"expected {REFERENCE_CONTINUOUS_STREAM_SHA256}, found {stream_sha256}."
        )
    chunk_bytes, count = checkpoint_layout(
        run_dir / "outhst", config["checkpoint_count"]
    )
    field_stream = run_dir / "fresh_fields.data"
    history = run_dir / "outhst"
    report = {
        "report_schema_version": GENERATION_SCHEMA_VERSION,
        "generator": generator_identity(),
        "script_version": SCRIPT_VERSION,
        "status": "integration_complete_pending_finalize",
        "configuration": config,
        "binding": {
            "preflight_report_file": preflight_path.name,
            "preflight_report_sha256": sha256_file(preflight_path),
            "source_manifest_sha256": preflight["build"]["upstream"][
                "local_source_manifest_sha256"
            ],
            "executable_sha256": sha256_file(executable),
        },
        "runtime": runtime,
        "field_stream_sha256": stream_sha256,
        "continuous_stream_reference_sha256": REFERENCE_CONTINUOUS_STREAM_SHA256,
        "continuous_stream_matches_pre_restart_audit_run": (
            stream_sha256 == REFERENCE_CONTINUOUS_STREAM_SHA256
        ),
        "reference_hash_enforced_for_recorded_toolchain": enforce_reference_hash,
        "reference_toolchain": REFERENCE_TOOLCHAIN,
        "history_sha256": sha256_file(run_dir / "outhst"),
        "checkpoint_chunk_bytes": chunk_bytes,
        "checkpoint_count": count,
        "artifacts": {
            "field_stream": {
                "file": field_stream.name,
                "size_bytes": field_stream.stat().st_size,
                "sha256": stream_sha256,
                "shape": list(fields.shape),
                "dtype": DTYPE.str,
                "layout": config["stream_layout"],
                "bytes_per_step": config["stream_bytes_per_step"],
            },
            "history": {
                "file": history.name,
                "size_bytes": history.stat().st_size,
                "sha256": sha256_file(history),
                "checkpoint_count": count,
                "checkpoint_chunk_bytes": chunk_bytes,
                "layout": "concatenated fixed-size native unformatted checkpoints",
            },
        },
    }
    write_json(workspace / "production_report.json", report)
    return report


def validate_file_record(path: Path, record: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    require_equal(path.stat().st_size, record["size_bytes"], f"{label} size")
    require_equal(sha256_file(path), record["sha256"], f"{label} SHA-256")


def validate_production_binding(
    args: argparse.Namespace, workspace: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    production_path = workspace / "production_report.json"
    if not production_path.is_file():
        raise RuntimeError("A production report is required before finalize")
    production = json.loads(production_path.read_text())
    require_equal(
        production.get("status"),
        "integration_complete_pending_finalize",
        "production status",
    )
    require_equal(
        production.get("report_schema_version"),
        GENERATION_SCHEMA_VERSION,
        "production report schema",
    )
    require_equal(
        production.get("generator"), generator_identity(), "production generator"
    )
    config = production_configuration(args)
    require_equal(production.get("configuration"), config, "finalize configuration")

    preflight, preflight_path = validate_preflight_binding(args, workspace)
    binding = production["binding"]
    require_equal(
        sha256_file(preflight_path),
        binding["preflight_report_sha256"],
        "bound preflight report SHA-256",
    )
    require_equal(
        preflight["build"]["upstream"]["local_source_manifest_sha256"],
        binding["source_manifest_sha256"],
        "bound source manifest",
    )
    executable = workspace / "source" / "zeqfc1"
    require_equal(
        sha256_file(executable), binding["executable_sha256"], "bound executable"
    )

    run_dir = workspace / "runs" / "production"
    field_stream = run_dir / production["artifacts"]["field_stream"]["file"]
    history = run_dir / production["artifacts"]["history"]["file"]
    stream_record = production["artifacts"]["field_stream"]
    history_record = production["artifacts"]["history"]
    expected_stream_size = config["retained_steps"] * config["stream_bytes_per_step"]
    require_equal(
        stream_record["size_bytes"], expected_stream_size, "stream recorded size"
    )
    require_equal(
        stream_record["shape"],
        [config["retained_steps"], len(FIELDS), N_LATITUDE, N_LONGITUDE],
        "stream recorded shape",
    )
    require_equal(stream_record["dtype"], DTYPE.str, "stream dtype")
    require_equal(stream_record["layout"], config["stream_layout"], "stream layout")
    require_equal(
        stream_record["bytes_per_step"],
        config["stream_bytes_per_step"],
        "stream bytes per step",
    )
    validate_file_record(field_stream, stream_record, "production field stream")
    validate_file_record(history, history_record, "production history")
    require_equal(
        history_record["checkpoint_count"], config["checkpoint_count"], "history count"
    )
    require_equal(
        history_record["size_bytes"],
        history_record["checkpoint_count"] * history_record["checkpoint_chunk_bytes"],
        "history layout size",
    )
    chunk_bytes, count = checkpoint_layout(history, config["checkpoint_count"])
    require_equal(
        chunk_bytes, history_record["checkpoint_chunk_bytes"], "history chunk"
    )
    require_equal(count, history_record["checkpoint_count"], "history checkpoint count")
    require_equal(
        production["field_stream_sha256"], stream_record["sha256"], "stream report hash"
    )
    require_equal(
        production["history_sha256"], history_record["sha256"], "history report hash"
    )
    return production, preflight, production_path, field_stream, history


def atomic_publish_directory(staging: Path, destination: Path, overwrite: bool) -> None:
    """Publish a complete sibling staging tree and roll back a failed swap."""

    if staging.parent != destination.parent:
        raise RuntimeError(
            "Staging and destination must be siblings for atomic publish"
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {destination}")
    backup: Path | None = None
    if destination.exists():
        backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def convert_fields(fields: np.ndarray, output_dir: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for index, definition in enumerate(FIELDS):
        path = output_dir / f"{definition['name']}.npy"
        target = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=DTYPE,
            shape=(fields.shape[0], N_LATITUDE, N_LONGITUDE),
        )
        chunk = 4096
        minimum = math.inf
        maximum = -math.inf
        for start in range(0, fields.shape[0], chunk):
            stop = min(start + chunk, fields.shape[0])
            values = np.asarray(fields[start:stop, index])
            if not np.isfinite(values).all():
                raise RuntimeError(f"Nonfinite values found in {definition['name']}")
            target[start:stop] = values
            minimum = min(minimum, float(np.min(values)))
            maximum = max(maximum, float(np.max(values)))
        target.flush()
        del target
        files[definition["name"]] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "minimum": minimum,
            "maximum": maximum,
        }
    return files


def nino3_stationarity_diagnostic(
    values: np.ndarray, steps_per_year: int
) -> dict[str, Any]:
    """Summarize, but do not gate on, post-spinup distribution stability."""

    window = 100 * steps_per_year
    if values.size < 3 * window:
        return {
            "status": "not_evaluated",
            "reason": "At least 300 retained years are required for the diagnostic.",
        }

    def summary(sample: np.ndarray) -> dict[str, Any]:
        return {
            "count": int(sample.size),
            "mean_c": float(np.mean(sample, dtype=np.float64)),
            "standard_deviation_c": float(np.std(sample, dtype=np.float64)),
            "quantiles_c": {
                str(probability): float(np.quantile(sample, probability))
                for probability in (0.01, 0.05, 0.5, 0.95, 0.99)
            },
            "fraction_above_0_5c": float(np.mean(sample >= 0.5)),
            "fraction_below_minus_0_5c": float(np.mean(sample <= -0.5)),
        }

    early = np.asarray(values[:window], dtype=np.float64)
    middle_start = values.size // 2 - window // 2
    middle = np.asarray(values[middle_start : middle_start + window], dtype=np.float64)
    late = np.asarray(values[-window:], dtype=np.float64)
    early_summary = summary(early)
    late_summary = summary(late)
    pooled = math.sqrt(
        0.5
        * (
            early_summary["standard_deviation_c"] ** 2
            + late_summary["standard_deviation_c"] ** 2
        )
    )
    return {
        "status": "reported_not_used_as_a_pass_fail_gate",
        "window_years": 100,
        "early_retained": early_summary,
        "middle_retained": summary(middle),
        "late_retained": late_summary,
        "early_late_absolute_mean_difference_c": abs(
            early_summary["mean_c"] - late_summary["mean_c"]
        ),
        "early_late_mean_difference_in_pooled_standard_deviations": (
            abs(early_summary["mean_c"] - late_summary["mean_c"]) / pooled
            if pooled > 0.0
            else None
        ),
        "early_late_standard_deviation_ratio": (
            early_summary["standard_deviation_c"] / late_summary["standard_deviation_c"]
            if late_summary["standard_deviation_c"] > 0.0
            else None
        ),
        "interpretation": (
            "This descriptive check compares the first, middle, and last 100 "
            "retained years after the separately discarded spinup. It is recorded "
            "for transparency and deliberately does not tune or reject the run."
        ),
    }


def event_checkpoint_filename(label: str, lead_steps: int) -> str:
    lead_months, remainder = divmod(lead_steps, STEPS_PER_MONTH)
    if remainder:
        raise RuntimeError("Event lead must be an integer number of model months")
    return f"{label}_{lead_months}month_pre_input.hst"


def event_checkpoint(
    *,
    label: str,
    target_index: int,
    lead_steps: int,
    config: dict[str, Any],
    workspace: Path,
    executable: Path,
    history: Path,
    checkpoint_chunk_bytes: int,
    fields: np.ndarray,
) -> dict[str, Any]:
    input_index = target_index - lead_steps
    if input_index < 0:
        raise RuntimeError(
            f"{label} target occurs before the requested lead is available"
        )
    input_nt = config["spinup_steps"] + 1 + input_index
    pre_input_nt = input_nt - 1
    checkpoint_steps = config["checkpoint_steps"]
    sparse_nt = (pre_input_nt // checkpoint_steps) * checkpoint_steps
    if sparse_nt == 0:
        raise RuntimeError("Selected event predates the first sparse checkpoint")
    sparse_index = sparse_nt // checkpoint_steps - 1
    event_dir = workspace / "event_checkpoints"
    event_dir.mkdir(exist_ok=True)
    sparse_file = event_dir / f"{label}_sparse_source.hst"
    extract_checkpoint(history, sparse_file, sparse_index, checkpoint_chunk_bytes)

    exact_file = event_dir / event_checkpoint_filename(label, lead_steps)
    if sparse_nt == pre_input_nt:
        shutil.copy2(sparse_file, exact_file)
    else:
        replay_dir = workspace / "runs" / f"select_{label}_checkpoint"
        sparse_time = model_time(sparse_nt)
        exact_time = model_time(pre_input_nt)
        prepare_run(
            workspace / "source",
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
            restart=sparse_file,
        )
        run_model(replay_dir)
        shutil.copy2(replay_dir / "outhst", exact_file)

    verify_dir = workspace / "runs" / f"verify_{label}_checkpoint"
    pre_time = model_time(pre_input_nt)
    target_time = model_time(input_nt + lead_steps)
    prepare_run(
        workspace / "source",
        executable,
        verify_dir,
        nstart=3,
        tfind=pre_time,
        tzero=pre_time,
        tend=target_time,
        ntape=0,
        nrewnd=11,
        nic=0,
        write_start=pre_time + 0.1,
        write_end=target_time,
        restart=exact_file,
    )
    run_model(verify_dir)
    replay = fresh_memmap(verify_dir / "fresh_fields.data")
    expected = fields[input_index : target_index + 1]
    equal = bool(np.array_equal(replay, expected))
    maximum_error = float(
        np.max(np.abs(replay.astype(np.float64) - expected.astype(np.float64)))
    )
    if not equal:
        raise RuntimeError(
            f"{label} checkpoint replay failed: max error {maximum_error}"
        )
    return {
        "label": label,
        "checkpoint_file": exact_file.name,
        "checkpoint_sha256": sha256_file(exact_file),
        "checkpoint_size_bytes": exact_file.stat().st_size,
        "checkpoint_semantics": (
            "Complete native state immediately before the event input update; "
            "the first restarted compact output is exactly event input_index."
        ),
        "input_index": input_index,
        "target_index": target_index,
        "lead_steps": lead_steps,
        "input_nt": input_nt,
        "pre_input_checkpoint_nt": pre_input_nt,
        "input_native_time_months": model_time(input_nt),
        "target_native_time_months": model_time(input_nt + lead_steps),
        "replay_steps": int(replay.shape[0]),
        "all_13_fields_bitwise_equal": equal,
        "maximum_absolute_error": maximum_error,
        "event_replay_executable_sha256": sha256_file(executable),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite: {output_dir}")
    production, preflight, production_path, field_stream, history = (
        validate_production_binding(args, workspace)
    )
    config = production["configuration"]
    fields = fresh_memmap(field_stream)
    require_equal(fields.shape[0], config["retained_steps"], "retained stream steps")

    nino3 = nino3_from_fields(fields)
    stationarity = nino3_stationarity_diagnostic(nino3, STEPS_PER_YEAR)
    lead_steps = config["event_lead_months"] * STEPS_PER_MONTH
    if config["retained_years"] != 12_000:
        raise RuntimeError(
            "Canonical event selection requires the 12,000-year retained data set."
        )
    test_start, test_stop = 11_000 * STEPS_PER_YEAR, 12_000 * STEPS_PER_YEAR
    eligible = np.arange(test_start + lead_steps, test_stop)
    warm_index = int(eligible[np.argmax(nino3[eligible])])
    cold_index = int(eligible[np.argmin(nino3[eligible])])
    executable = workspace / "source" / "zeqfc1"
    event_results = []
    for label, index in (
        ("extreme_el_nino", warm_index),
        ("extreme_la_nina", cold_index),
    ):
        event_results.append(
            event_checkpoint(
                label=label,
                target_index=index,
                lead_steps=lead_steps,
                config=config,
                workspace=workspace,
                executable=executable,
                history=history,
                checkpoint_chunk_bytes=production["artifacts"]["history"][
                    "checkpoint_chunk_bytes"
                ],
                fields=fields,
            )
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        field_files = convert_fields(fields, staging)
        np.save(staging / "nino3_index.npy", nino3, allow_pickle=False)
        first_nt = config["spinup_steps"] + 1
        native_time = np.asarray(
            [model_time(first_nt + index) for index in range(fields.shape[0])],
            dtype=np.float64,
        )
        np.save(staging / "native_time_months.npy", native_time, allow_pickle=False)
        phase = phase_features(native_time)
        np.save(staging / "annual_phase_sin_cos.npy", phase, allow_pickle=False)
        auxiliary_files = {}
        for filename in (
            "nino3_index.npy",
            "native_time_months.npy",
            "annual_phase_sin_cos.npy",
        ):
            path = staging / filename
            auxiliary_files[filename] = {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        split = {
            "train": [0, 10_000 * STEPS_PER_YEAR],
            "validation": [10_000 * STEPS_PER_YEAR, 11_000 * STEPS_PER_YEAR],
            "test": [11_000 * STEPS_PER_YEAR, 12_000 * STEPS_PER_YEAR],
            "semantics": (
                "half-open chronological indices before lead-specific pair construction"
            ),
            "normalization": (
                "fit means/scales for every selected spatial coordinate and both phase "
                "scalars on all raw states in the 10000-year train block only"
            ),
        }
        for event in event_results:
            source = workspace / "event_checkpoints" / event["checkpoint_file"]
            destination = staging / event["checkpoint_file"]
            shutil.copy2(source, destination)

        build = preflight["build"]
        upstream = build["upstream"]
        compiler = build["compiler"]
        release_provenance = {
            "generation_schema_version": GENERATION_SCHEMA_VERSION,
            "generator": generator_identity(),
            "report_bindings": {
                "preflight_report_file": "preflight_report.json",
                "preflight_report_sha256": sha256_file(
                    workspace / "preflight_report.json"
                ),
                "production_report_file": production_path.name,
                "production_report_sha256": sha256_file(production_path),
            },
            "upstream": {
                "download_url": upstream["download_url"],
                "expected_zip_sha256": upstream["expected_zip_sha256"],
                "source_directory_name": Path(upstream["local_source_directory"]).name,
                "source_manifest_sha256": upstream["local_source_manifest_sha256"],
                "source_files": upstream["local_source_files"],
                "redistributed_with_processed_data": False,
            },
            "patch": build["patch"],
            "build": {
                "compiler_program": Path(compiler["compiler"]).name,
                "compiler_version": compiler["compiler_version"],
                "make_program": Path(compiler["make"]).name,
                "platform": compiler["platform"],
                "machine": compiler["machine"],
                "byteorder": compiler["byteorder"],
                "flags": build["flags"],
                "link_flags": (
                    "-isysroot <macOS SDK>" if compiler.get("sdk_path") else ""
                ),
                "executable_sha256": build["executable_sha256"],
            },
            "production_configuration": config,
            "native_artifacts": production["artifacts"],
            "reference_stream": {
                "sha256": REFERENCE_CONTINUOUS_STREAM_SHA256,
                "recorded_toolchain": REFERENCE_TOOLCHAIN,
                "exact_hash_enforced_for_this_run": production[
                    "reference_hash_enforced_for_recorded_toolchain"
                ],
                "matches": production[
                    "continuous_stream_matches_pre_restart_audit_run"
                ],
            },
            "restart_certification": preflight["restart"],
        }
        provenance_path = staging / "generation_provenance.json"
        write_json(provenance_path, release_provenance)
        provenance_file = {
            "file": provenance_path.name,
            "size_bytes": provenance_path.stat().st_size,
            "sha256": sha256_file(provenance_path),
        }
        metadata = {
            "schema_version": DATA_SCHEMA_VERSION,
            "generation_schema_version": GENERATION_SCHEMA_VERSION,
            "script_version": SCRIPT_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "provenance": {
                "generation_binding": provenance_file,
                "upstream_download_url": upstream["download_url"],
                "upstream_archive_sha256": upstream["expected_zip_sha256"],
                "source_manifest_sha256": upstream["local_source_manifest_sha256"],
                "executable_sha256": build["executable_sha256"],
            },
            "integration": {
                "spinup_years_discarded": config["spinup_years"],
                "retained_years": config["retained_years"],
                "checkpoint_years": config["checkpoint_years"],
                "event_lead_months": config["event_lead_months"],
                "steps_per_month": config["steps_per_month"],
                "steps_per_year": config["steps_per_year"],
                "retained_steps": config["retained_steps"],
                "native_time_units": "months; 0.5 is mid-January of nominal year 1960",
                "native_time_file": "native_time_months.npy",
            },
            "grid": {
                "shape": [N_LATITUDE, N_LONGITUDE],
                "latitude_degrees_north": LATITUDES.tolist(),
                "longitude_degrees_east": LONGITUDES.tolist(),
                "active_fortran_indices_one_based": {
                    "ordinary_fields": {
                        "latitude": [25, 6, -1],
                        "longitude": [6, 32, 1],
                    },
                    "HTAU_latitude_mapping": "HTAU longitude J, latitude 31-I",
                },
            },
            "fields": list(FIELDS),
            "field_files": field_files,
            "auxiliary_files": auxiliary_files,
            "model_input_variants": {
                "primary_core_plus_phase": {
                    "spatial_fields": list(CORE_FIELDS),
                    "phase_features": ["annual_phase_sin", "annual_phase_cos"],
                },
                "core_plus_upwelling_plus_phase": {
                    "spatial_fields": list(CORE_FIELDS + ("upwelling_anomaly",)),
                    "phase_features": ["annual_phase_sin", "annual_phase_cos"],
                },
                "legacy_ten_plus_phase": {
                    "spatial_fields": list(LEGACY_TEN_FIELDS),
                    "phase_features": ["annual_phase_sin", "annual_phase_cos"],
                },
                "all_interpretable_plus_phase": {
                    "spatial_fields": [field["name"] for field in FIELDS],
                    "phase_features": ["annual_phase_sin", "annual_phase_cos"],
                },
            },
            "phase": {
                "file": "annual_phase_sin_cos.npy",
                "shape": [int(fields.shape[0]), 2],
                "formula": (
                    "angle=2*pi*mod(native_time_months-0.5,12)/12; "
                    "columns sin(angle),cos(angle)"
                ),
                "storage": (
                    "two scalars per time, never duplicated over the spatial grid"
                ),
            },
            "nino3": {
                "file": "nino3_index.npy",
                "definition": (
                    "unweighted mean of model-grid SST-anomaly centers inside or "
                    "on 5S-5N, 150W-90W"
                ),
                "latitude_degrees_north": LATITUDES[NINO3_LATITUDE_MASK].tolist(),
                "longitude_degrees_east": LONGITUDES[NINO3_LONGITUDE_MASK].tolist(),
                "grid_point_count": int(
                    NINO3_LATITUDE_MASK.sum() * NINO3_LONGITUDE_MASK.sum()
                ),
                "includes_270E_90W_center": True,
            },
            "spinup_stationarity_diagnostic": stationarity,
            "chronological_split": split,
            "event_restart_checkpoints": event_results,
            "validation": {
                "preflight_report_sha256": sha256_file(
                    workspace / "preflight_report.json"
                ),
                "compact_first_ten_match_legacy_output": True,
                "extended_restart_all_13_fields_bitwise_equal": True,
                "event_restarts_bitwise_equal": all(
                    item["all_13_fields_bitwise_equal"] for item in event_results
                ),
                "restart_scope": (
                    "Certified for the distributed Standard configuration: no WWB "
                    "build, mask_heating false, seasonal-background freeze disabled, "
                    "mid-run SST dissipation change disabled, and NIC=0 on restart. "
                    "Other compile-time "
                    "or namelist branches may contain additional persistent state."
                ),
            },
            "release_note": (
                "The upstream ZC source is not included. Download it from the "
                "recorded URL, verify the ZIP SHA-256, and run this script. "
                "Processed arrays and the two "
                "derived native event checkpoints are intended for the Zenodo release."
            ),
        }
        metadata_path = staging / "metadata.json"
        write_json(metadata_path, metadata)

        for record in field_files.values():
            validate_file_record(staging / record["file"], record, record["file"])
        for filename, record in auxiliary_files.items():
            validate_file_record(staging / filename, record, filename)
        validate_file_record(provenance_path, provenance_file, provenance_path.name)
        for event in event_results:
            event_path = staging / event["checkpoint_file"]
            require_equal(
                event_path.stat().st_size,
                event["checkpoint_size_bytes"],
                f"{event_path.name} size",
            )
            require_equal(
                sha256_file(event_path),
                event["checkpoint_sha256"],
                f"{event_path.name} SHA-256",
            )
        metadata_sha256 = sha256_file(metadata_path)
        atomic_publish_directory(staging, output_dir, args.overwrite)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise

    del fields
    if not args.keep_native_field_stream:
        field_stream.unlink()
    final = {
        "report_schema_version": GENERATION_SCHEMA_VERSION,
        "generator": generator_identity(),
        "status": "complete",
        "output_directory": str(output_dir),
        "metadata_sha256": metadata_sha256,
        "production_report_sha256": sha256_file(production_path),
        "extreme_el_nino_c": float(nino3[warm_index]),
        "extreme_la_nina_c": float(nino3[cold_index]),
        "events": event_results,
    }
    write_json(workspace / "finalize_report.json", final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "preflight":
        result = preflight(args)
    elif args.mode == "generate":
        result = generate(args)
    else:
        result = finalize(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
