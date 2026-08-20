import argparse
import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from autonomy_sim.experiments.run_campaign import (
    CampaignConfig,
    get_git_state,
    load_campaign_config,
    run_campaign,
)
from autonomy_sim.experiments.run_monte_carlo import TRIAL_FIELDS
from autonomy_sim.sensors.sensor_scenarios import sensor_scenario_name


SENSOR_SCENARIOS = (0, 1, 2, 3, 4)
PAIRED_METRICS = (
    "completion_time",
    "final_true_distance",
    "rmse_estimation_error",
    "rmse_true_cross_track_error",
    "minimum_true_clearance",
    "applied_control_effort",
    "measurement_dropout_fraction",
)


@dataclass
class SensorComparisonConfig:
    source_path: Path
    name: str
    frozen: bool
    primary_sensor_scenario: int
    campaign_configs: list[CampaignConfig]
    output_dir: Path


def load_sensor_comparison_config(path) -> SensorComparisonConfig:
    source_path = Path(path).resolve()
    with source_path.open(encoding="utf-8") as config_file:
        settings = yaml.safe_load(config_file)
    if not isinstance(settings, dict):
        raise ValueError("sensor comparison configuration root must be a mapping")

    required_fields = {
        "type",
        "name",
        "frozen",
        "primary_sensor_scenario",
        "campaign_configs",
        "output_dir",
    }
    if set(settings) != required_fields:
        missing = sorted(required_fields - set(settings))
        unknown = sorted(set(settings) - required_fields)
        raise ValueError(
            "sensor comparison fields must match the schema; "
            f"missing={missing}, unknown={unknown}"
        )
    if settings["type"] != "sensor_comparison_campaign":
        raise ValueError("sensor comparison type must be 'sensor_comparison_campaign'")

    name = settings["name"]
    if not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) is None:
        raise ValueError(
            "sensor comparison name must use lowercase letters, numbers, '-' or '_'"
        )
    frozen = settings["frozen"]
    if not isinstance(frozen, bool):
        raise ValueError("sensor comparison frozen must be boolean")
    primary_sensor_scenario = settings["primary_sensor_scenario"]
    if primary_sensor_scenario not in SENSOR_SCENARIOS:
        raise ValueError("primary_sensor_scenario must be between zero and four")

    campaign_names = settings["campaign_configs"]
    if not isinstance(campaign_names, list) or not campaign_names:
        raise ValueError("campaign_configs must be a non-empty list")
    parent = source_path.parent
    campaign_configs = [
        load_campaign_config(parent / campaign_name) for campaign_name in campaign_names
    ]
    output_dir = (parent / settings["output_dir"]).resolve()
    config = SensorComparisonConfig(
        source_path=source_path,
        name=name,
        frozen=frozen,
        primary_sensor_scenario=primary_sensor_scenario,
        campaign_configs=campaign_configs,
        output_dir=output_dir,
    )
    _validate_campaign_configs(config)
    return config


def run_sensor_comparison(config: SensorComparisonConfig):
    if config.output_dir.exists():
        raise FileExistsError(
            f"sensor comparison output already exists: {config.output_dir}"
        )
    source_git_state = get_git_state()
    if config.frozen and source_git_state["dirty"]:
        raise RuntimeError("a frozen comparison requires a clean Git working tree")

    rows_by_scenario = {}
    summaries = {}
    qualifications = {}
    manifests = {}
    for campaign_config in config.campaign_configs:
        manifest, summary, qualification = run_campaign(
            campaign_config,
            source_git_state=source_git_state,
        )
        scenario = campaign_config.sensor_scenario
        rows_by_scenario[scenario] = _read_rows(
            campaign_config.output_dir / "trials.csv"
        )
        summaries[scenario] = summary
        qualifications[scenario] = qualification
        manifests[scenario] = manifest

    _validate_matched_rows(rows_by_scenario)
    comparison = build_comparison(
        rows_by_scenario,
        summaries,
        qualifications,
        primary_sensor_scenario=config.primary_sensor_scenario,
    )
    _write_paired_trials(config.output_dir / "paired_trials.csv", rows_by_scenario)
    _write_json(config.output_dir / "comparison.json", comparison)
    plot_sensor_comparison(
        comparison,
        config.output_dir / "sensor_comparison_summary.png",
    )
    manifest = _write_comparison_manifest(config, manifests, qualifications)
    return manifest, comparison


def build_comparison(
    rows_by_scenario,
    summaries,
    qualifications,
    *,
    primary_sensor_scenario,
):
    nominal_rows = rows_by_scenario[0]
    scenario_results = {}
    comparisons_to_nominal = {}
    for scenario in SENSOR_SCENARIOS:
        name = sensor_scenario_name(scenario)
        scenario_results[name] = {
            "sensor_scenario": scenario,
            "summary": summaries[scenario],
            "qualification": qualifications[scenario],
        }
        if scenario != 0:
            comparisons_to_nominal[name] = _compare_matched_rows(
                nominal_rows,
                rows_by_scenario[scenario],
                summaries[0],
                summaries[scenario],
            )

    return {
        "configuration": {
            "trials_per_scenario": len(nominal_rows),
            "total_trials": len(nominal_rows) * len(SENSOR_SCENARIOS),
            "sensor_scenarios": list(SENSOR_SCENARIOS),
        },
        "primary_research_scenario": {
            "sensor_scenario": primary_sensor_scenario,
            "sensor_scenario_name": sensor_scenario_name(primary_sensor_scenario),
        },
        "scenarios": scenario_results,
        "comparisons_to_nominal": comparisons_to_nominal,
    }


def plot_sensor_comparison(comparison, output_path):
    plt.switch_backend("Agg")
    names = [sensor_scenario_name(scenario) for scenario in SENSOR_SCENARIOS]
    summaries = [comparison["scenarios"][name]["summary"] for name in names]
    short_names = ("nominal", "offset", "delay", "dropout", "combined")

    mission_success = [
        summary["rates"]["true_mission_success_given_plan"] or 0.0
        for summary in summaries
    ]
    estimation_rmse = [
        summary["statistics"]["rmse_estimation_error"]["mean"] or 0.0
        for summary in summaries
    ]
    cross_track_rmse = [
        summary["statistics"]["rmse_true_cross_track_error"]["mean"] or 0.0
        for summary in summaries
    ]
    dropout_fraction = [
        summary["statistics"]["measurement_dropout_fraction"]["mean"] or 0.0
        for summary in summaries
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    _bar(axes[0, 0], short_names, mission_success, "Mission success | plan")
    axes[0, 0].set_ylim(0.0, 1.0)
    _bar(axes[0, 1], short_names, estimation_rmse, "Estimation RMSE")
    _bar(axes[1, 0], short_names, cross_track_rmse, "True cross-track RMSE")
    _bar(axes[1, 1], short_names, dropout_fraction, "Measurement dropout fraction")
    axes[1, 1].set_ylim(0.0, 1.0)
    figure.suptitle("Matched sensor-scenario comparison")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _validate_campaign_configs(config):
    scenarios = [campaign.sensor_scenario for campaign in config.campaign_configs]
    if sorted(scenarios) != list(SENSOR_SCENARIOS):
        raise ValueError("comparison must contain each sensor scenario exactly once")

    reference = config.campaign_configs[0]
    for campaign in config.campaign_configs:
        if campaign.frozen != config.frozen:
            raise ValueError("comparison and child campaign frozen settings must match")
        if campaign.trials != reference.trials:
            raise ValueError("comparison campaigns must use the same trial count")
        if campaign.base_seed != reference.base_seed:
            raise ValueError("comparison campaigns must use the same base seed")
        if campaign.simulation_config_path != reference.simulation_config_path:
            raise ValueError("comparison campaigns must use the same simulation input")
        if campaign.mission_config_path != reference.mission_config_path:
            raise ValueError("comparison campaigns must use the same mission input")
        if campaign.output_dir.parent != config.output_dir:
            raise ValueError(
                "campaign outputs must be direct comparison subdirectories"
            )


def _validate_matched_rows(rows_by_scenario):
    nominal_rows = rows_by_scenario[0]
    for scenario in SENSOR_SCENARIOS[1:]:
        degraded_rows = rows_by_scenario[scenario]
        if len(degraded_rows) != len(nominal_rows):
            raise RuntimeError("sensor scenarios produced different trial counts")
        for nominal, degraded in zip(nominal_rows, degraded_rows, strict=True):
            for field in ("trial", "environment_seed", "sensor_seed"):
                if nominal[field] != degraded[field]:
                    raise RuntimeError(
                        f"sensor scenarios are not paired at field {field}"
                    )
            if nominal["planning_success"] != degraded["planning_success"]:
                raise RuntimeError("sensor scenario changed the planning outcome")


def _compare_matched_rows(
    nominal_rows,
    degraded_rows,
    nominal_summary,
    degraded_summary,
):
    planned_pairs = [
        (nominal, degraded)
        for nominal, degraded in zip(nominal_rows, degraded_rows, strict=True)
        if nominal["planning_success"]
    ]
    outcome_transitions = {
        "both_succeeded": 0,
        "nominal_only_succeeded": 0,
        "degraded_only_succeeded": 0,
        "neither_succeeded": 0,
    }
    for nominal, degraded in planned_pairs:
        nominal_success = nominal["true_mission_success"]
        degraded_success = degraded["true_mission_success"]
        if nominal_success and degraded_success:
            outcome_transitions["both_succeeded"] += 1
        elif nominal_success:
            outcome_transitions["nominal_only_succeeded"] += 1
        elif degraded_success:
            outcome_transitions["degraded_only_succeeded"] += 1
        else:
            outcome_transitions["neither_succeeded"] += 1

    rate_deltas = {}
    for field in (
        "true_mission_success_given_plan",
        "false_completion_given_plan",
        "safety_violation_given_plan",
    ):
        nominal_value = nominal_summary["rates"][field]
        degraded_value = degraded_summary["rates"][field]
        rate_deltas[field] = (
            None
            if nominal_value is None or degraded_value is None
            else float(degraded_value - nominal_value)
        )

    metric_deltas = {
        field: _paired_delta_statistics(planned_pairs, field)
        for field in PAIRED_METRICS
    }
    return {
        "planned_trial_pairs": len(planned_pairs),
        "mission_outcome_transitions": outcome_transitions,
        "rate_deltas_degraded_minus_nominal": rate_deltas,
        "paired_metric_deltas_degraded_minus_nominal": metric_deltas,
    }


def _paired_delta_statistics(pairs, field):
    deltas = []
    for nominal, degraded in pairs:
        nominal_value = nominal[field]
        degraded_value = degraded[field]
        if _is_finite_number(nominal_value) and _is_finite_number(degraded_value):
            deltas.append(float(degraded_value - nominal_value))
    if not deltas:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    values = np.array(deltas, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as csv_file:
        rows = []
        for row in csv.DictReader(csv_file):
            rows.append(_coerce_row(row))
        return rows


def _coerce_row(row):
    converted = {}
    boolean_fields = {
        "planning_success",
        "onboard_completion",
        "true_goal_reached",
        "true_mission_success",
        "restricted_violation",
        "disallowed_violation",
        "out_of_bounds_violation",
    }
    integer_fields = {"trial", "environment_seed", "sensor_seed", "sensor_scenario"}
    text_fields = {"sensor_scenario_name", "sensor_model", "termination_state"}
    for field, value in row.items():
        if field in boolean_fields:
            converted[field] = None if value == "" else value == "True"
        elif field in integer_fields:
            converted[field] = int(value)
        elif field in text_fields:
            converted[field] = value
        else:
            converted[field] = None if value == "" else float(value)
    return converted


def _write_paired_trials(path, rows_by_scenario):
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRIAL_FIELDS)
        writer.writeheader()
        for trial in range(len(rows_by_scenario[0])):
            for scenario in SENSOR_SCENARIOS:
                writer.writerow(rows_by_scenario[scenario][trial])


def _write_comparison_manifest(config, manifests, qualifications):
    inputs_dir = config.output_dir / "inputs"
    inputs_dir.mkdir(exist_ok=True)
    archived_config = inputs_dir / "comparison.yaml"
    shutil.copyfile(config.source_path, archived_config)
    first_manifest = manifests[0]
    configs_by_scenario = {
        campaign.sensor_scenario: campaign for campaign in config.campaign_configs
    }
    scenario_manifests = {}
    scenario_qualification = {}
    for scenario in SENSOR_SCENARIOS:
        name = sensor_scenario_name(scenario)
        manifest_path = configs_by_scenario[scenario].output_dir / "manifest.json"
        scenario_manifests[name] = str(manifest_path.relative_to(config.output_dir))
        scenario_qualification[name] = qualifications[scenario]["status"]

    manifest = {
        "schema_version": 1,
        "comparison": {
            "name": config.name,
            "frozen": config.frozen,
            "trials_per_scenario": first_manifest["campaign"]["trials"],
            "total_trials": first_manifest["campaign"]["trials"]
            * len(SENSOR_SCENARIOS),
            "base_seed": first_manifest["campaign"]["base_seed"],
            "primary_sensor_scenario": config.primary_sensor_scenario,
            "primary_sensor_scenario_name": sensor_scenario_name(
                config.primary_sensor_scenario
            ),
        },
        "code": first_manifest["code"],
        "runtime": first_manifest["runtime"],
        "input": {
            "archived": str(archived_config.relative_to(config.output_dir)),
            "sha256": _sha256(archived_config),
        },
        "scenario_manifests": scenario_manifests,
        "scenario_qualification": scenario_qualification,
        "artifacts": {
            "comparison": "comparison.json",
            "paired_trials": "paired_trials.csv",
            "plot": "sensor_comparison_summary.png",
        },
    }
    _write_json(config.output_dir / "manifest.json", manifest)
    return manifest


def _bar(axis, labels, values, title):
    axis.bar(labels, values)
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=20)


def _is_finite_number(value):
    return value is not None and not isinstance(value, bool) and np.isfinite(value)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run matched Monte Carlo campaigns for all sensor scenarios."
    )
    parser.add_argument("--comparison-config", type=Path, required=True)
    args = parser.parse_args()
    manifest, comparison = run_sensor_comparison(
        load_sensor_comparison_config(args.comparison_config)
    )
    print(
        json.dumps(
            {"manifest": manifest, "comparison": comparison},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
