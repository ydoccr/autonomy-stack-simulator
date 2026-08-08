import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from autonomy_sim.experiments.run_monte_carlo import (
    flatten_trial_result,
    run_monte_carlo,
)
from autonomy_sim.main import load_config
from autonomy_sim.mission.config import load_random_mission_config
from autonomy_sim.mission.run_random_sensor_scenarios import (
    run_random_sensor_scenario,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CRITERION_NAMES = (
    "minimum_planning_success_rate",
    "minimum_mission_success_given_plan_rate",
    "maximum_false_completion_given_plan_rate",
    "maximum_safety_violation_given_plan_rate",
    "maximum_mean_cross_track_rmse",
    "maximum_mean_control_saturation_fraction",
)


@dataclass
class CampaignConfig:
    source_path: Path
    name: str
    frozen: bool
    trials: int
    base_seed: int
    sensor_scenario: int
    workers: int
    simulation_config_path: Path
    mission_config_path: Path
    output_dir: Path
    criteria: dict[str, float]


def load_campaign_config(path) -> CampaignConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    if not isinstance(settings, dict):
        raise ValueError("campaign configuration root must be a mapping")

    required_fields = {
        "type",
        "name",
        "frozen",
        "trials",
        "base_seed",
        "sensor_scenario",
        "workers",
        "simulation_config",
        "mission_config",
        "output_dir",
        "criteria",
    }
    if set(settings) != required_fields:
        missing = sorted(required_fields - set(settings))
        unknown = sorted(set(settings) - required_fields)
        raise ValueError(
            f"campaign fields must match the schema; missing={missing}, unknown={unknown}"
        )
    if settings["type"] != "monte_carlo_campaign":
        raise ValueError("campaign type must be 'monte_carlo_campaign'")

    name = settings["name"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None:
        raise ValueError(
            "campaign name must use lowercase letters, numbers, '-' or '_'"
        )
    frozen = _boolean(settings, "frozen")
    trials = _integer(settings, "trials", minimum=1)
    base_seed = _integer(settings, "base_seed", minimum=0)
    sensor_scenario = _integer(settings, "sensor_scenario", minimum=0)
    if sensor_scenario > 4:
        raise ValueError("campaign sensor_scenario must be between zero and four")
    workers = _integer(settings, "workers", minimum=1)
    criteria = _criteria(settings["criteria"], frozen)
    parent = source_path.parent

    return CampaignConfig(
        source_path=source_path,
        name=name,
        frozen=frozen,
        trials=trials,
        base_seed=base_seed,
        sensor_scenario=sensor_scenario,
        workers=workers,
        simulation_config_path=(parent / settings["simulation_config"]).resolve(),
        mission_config_path=(parent / settings["mission_config"]).resolve(),
        output_dir=(parent / settings["output_dir"]).resolve(),
        criteria=criteria,
    )


def run_campaign(campaign_config: CampaignConfig, *, overwrite=False):
    git_state = _git_state()
    if campaign_config.frozen and git_state["dirty"]:
        raise RuntimeError("a frozen campaign requires a clean Git working tree")

    output_dir = campaign_config.output_dir
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"campaign output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True)
    archived_inputs = _archive_inputs(campaign_config, inputs_dir)

    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    rows, summary = run_monte_carlo(
        load_config(campaign_config.simulation_config_path),
        load_random_mission_config(campaign_config.mission_config_path),
        trials=campaign_config.trials,
        base_seed=campaign_config.base_seed,
        sensor_scenario=campaign_config.sensor_scenario,
        output_dir=output_dir,
        workers=campaign_config.workers,
    )
    duration_seconds = time.perf_counter() - start_time
    qualification = evaluate_qualification(summary, campaign_config.criteria)
    _write_json(output_dir / "qualification.json", qualification)
    plot_campaign_summary(rows, summary, output_dir / "qualification_summary.png")

    manifest = {
        "schema_version": 1,
        "campaign": {
            "name": campaign_config.name,
            "frozen": campaign_config.frozen,
            "trials": campaign_config.trials,
            "base_seed": campaign_config.base_seed,
            "sensor_scenario": campaign_config.sensor_scenario,
            "workers": campaign_config.workers,
        },
        "execution": {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": float(duration_seconds),
            "command": (
                "python -m autonomy_sim.experiments.run_campaign run "
                f"--campaign-config {campaign_config.source_path}"
            ),
        },
        "code": git_state,
        "runtime": _runtime_metadata(),
        "inputs": archived_inputs,
        "artifacts": {
            "trials": "trials.csv",
            "summary": "summary.json",
            "qualification": "qualification.json",
            "plot": "qualification_summary.png",
        },
        "qualification_passed": qualification["passed"],
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest, summary, qualification


def evaluate_qualification(summary, criteria):
    if not criteria:
        return {"status": "not_evaluated", "passed": None, "gates": {}}

    observations = {
        "minimum_planning_success_rate": summary["rates"]["planning_success"],
        "minimum_mission_success_given_plan_rate": summary["rates"][
            "true_mission_success_given_plan"
        ],
        "maximum_false_completion_given_plan_rate": summary["rates"][
            "false_completion_given_plan"
        ],
        "maximum_safety_violation_given_plan_rate": summary["rates"][
            "safety_violation_given_plan"
        ],
        "maximum_mean_cross_track_rmse": summary["statistics"][
            "rmse_true_cross_track_error"
        ]["mean"],
        "maximum_mean_control_saturation_fraction": summary["statistics"][
            "control_saturation_fraction"
        ]["mean"],
    }
    gates = {}
    for name in CRITERION_NAMES:
        observed = observations[name]
        threshold = criteria[name]
        is_minimum = name.startswith("minimum_")
        passed = observed is not None and (
            observed >= threshold if is_minimum else observed <= threshold
        )
        gates[name] = {
            "observed": observed,
            "operator": ">=" if is_minimum else "<=",
            "threshold": threshold,
            "passed": bool(passed),
        }
    overall_passed = all(gate["passed"] for gate in gates.values())
    return {
        "status": "passed" if overall_passed else "failed",
        "passed": overall_passed,
        "gates": gates,
    }


def plot_campaign_summary(rows, summary, output_path):
    plt.switch_backend("Agg")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    rate_names = (
        "planning_success",
        "true_mission_success_given_plan",
        "false_completion_given_plan",
        "safety_violation_given_plan",
    )
    rate_labels = ("Plan", "Mission | plan", "False completion | plan", "Safety | plan")
    rate_values = [summary["rates"][name] or 0.0 for name in rate_names]
    axes[0, 0].bar(rate_labels, rate_values)
    axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].set_title("Campaign outcome rates")
    axes[0, 0].tick_params(axis="x", rotation=20)

    termination_names = list(summary["termination_counts"])
    termination_values = [
        summary["termination_counts"][name] for name in termination_names
    ]
    axes[0, 1].bar(termination_names, termination_values)
    axes[0, 1].set_ylabel("Trials")
    axes[0, 1].set_title("Termination states")
    axes[0, 1].tick_params(axis="x", rotation=20)

    _histogram(
        axes[1, 0],
        rows,
        "rmse_true_cross_track_error",
        "True cross-track RMSE",
    )
    _histogram(
        axes[1, 1],
        rows,
        "control_saturation_fraction",
        "Control saturation fraction",
    )
    figure.suptitle("Nominal baseline qualification")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def replay_trial(manifest_path, trial, *, output_path=None):
    manifest_path = Path(manifest_path).resolve()
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    output_dir = manifest_path.parent
    expected = _read_trial(output_dir / manifest["artifacts"]["trials"], trial)
    simulation_config = load_config(output_dir / "inputs" / "simulation.yaml")
    mission_config = load_random_mission_config(output_dir / "inputs" / "mission.yaml")
    scenario = int(expected["sensor_scenario"])
    environment_seed = int(expected["environment_seed"])
    sensor_seed = int(expected["sensor_seed"])
    result, *_ = run_random_sensor_scenario(
        simulation_config,
        mission_config,
        scenario,
        environment_seed,
        sensor_seed,
        show_plots=False,
        show_metrics=False,
    )
    actual = flatten_trial_result(
        trial,
        environment_seed,
        sensor_seed,
        scenario,
        result,
    )
    mismatches = {}
    for name, actual_value in actual.items():
        expected_value = _coerce_expected(expected[name], actual_value)
        if not _values_match(expected_value, actual_value):
            mismatches[name] = {"expected": expected_value, "actual": actual_value}
    report = {
        "trial": int(trial),
        "matched": not mismatches,
        "mismatches": mismatches,
    }
    if output_path is not None:
        _write_json(Path(output_path), report)
    return report


def _archive_inputs(config, inputs_dir):
    sources = {
        "campaign": config.source_path,
        "simulation": config.simulation_config_path,
        "mission": config.mission_config_path,
    }
    archived_names = {
        "campaign": "campaign.yaml",
        "simulation": "simulation.yaml",
        "mission": "mission.yaml",
    }
    metadata = {}
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"campaign input does not exist: {source}")
        archived_path = inputs_dir / archived_names[name]
        shutil.copyfile(source, archived_path)
        metadata[name] = {
            "source": str(source),
            "archived": str(Path("inputs") / archived_path.name),
            "sha256": _sha256(archived_path),
        }
    return metadata


def _git_state():
    commit = _git_command("rev-parse", "HEAD")
    status = _git_command("status", "--porcelain", "--untracked-files=all")
    return {"commit": commit, "dirty": bool(status)}


def _git_command(*arguments):
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _runtime_metadata():
    packages = {}
    for package in ("matplotlib", "numpy", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def _criteria(value, frozen):
    if not isinstance(value, dict):
        raise ValueError("campaign criteria must be a mapping")
    if not value:
        if frozen:
            raise ValueError("a frozen campaign requires qualification criteria")
        return {}
    if set(value) != set(CRITERION_NAMES):
        raise ValueError(
            "campaign criteria must define the complete qualification schema"
        )
    criteria = {
        name: _finite_number(value[name], f"criteria.{name}")
        for name in CRITERION_NAMES
    }
    for name, threshold in criteria.items():
        is_cross_track = name == "maximum_mean_cross_track_rmse"
        if is_cross_track and threshold < 0.0:
            raise ValueError(f"campaign {name} must be non-negative")
        if not is_cross_track and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"campaign {name} must be between zero and one")
    return criteria


def _boolean(settings, name):
    value = settings[name]
    if not isinstance(value, bool):
        raise ValueError(f"campaign {name} must be boolean")
    return value


def _integer(settings, name, minimum):
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"campaign {name} must be an integer >= {minimum}")
    return value


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"campaign {name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"campaign {name} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"campaign {name} must be finite")
    return number


def _histogram(ax, rows, field, title):
    values = [float(row[field]) for row in rows if _is_finite(row[field])]
    if values:
        ax.hist(values, bins=min(20, max(5, round(np.sqrt(len(values))))))
    ax.set_title(title)
    ax.set_ylabel("Trials")


def _is_finite(value):
    return value is not None and not isinstance(value, bool) and np.isfinite(value)


def _read_trial(path, trial):
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            if int(row["trial"]) == int(trial):
                return row
    raise ValueError(f"trial {trial} is not present in {path}")


def _coerce_expected(value, actual):
    if actual is None:
        return None if value == "" else value
    if isinstance(actual, bool):
        return value == "True"
    if isinstance(actual, int):
        return int(value)
    if isinstance(actual, float):
        return float(value)
    return value


def _values_match(expected, actual):
    if isinstance(actual, float):
        if np.isnan(actual):
            return isinstance(expected, float) and np.isnan(expected)
        return bool(np.isclose(expected, actual, rtol=1e-12, atol=1e-12))
    return expected == actual


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Run or replay a campaign.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--campaign-config", type=Path, required=True)
    run_parser.add_argument("--overwrite", action="store_true")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--manifest", type=Path, required=True)
    replay_parser.add_argument("--trial", type=int, required=True)
    replay_parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        manifest, summary, qualification = run_campaign(
            load_campaign_config(args.campaign_config),
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "manifest": manifest,
                    "summary": summary,
                    "qualification": qualification,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        report = replay_trial(
            args.manifest,
            args.trial,
            output_path=args.output,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["matched"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
