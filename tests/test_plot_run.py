import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from autonomy_sim.core.types import Waypoint
from autonomy_sim.metrics.metrics_run import RunMetrics
from autonomy_sim.visualization.plot_run import plot_all, plot_metrics
from test_run_metrics import create_trajectory


def test_plot_all_uses_one_dashboard_window(monkeypatch):
    plt.close("all")
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))

    trajectory = create_trajectory()
    waypoints = [Waypoint(x=2.0, y=0.0)]
    metrics = RunMetrics(trajectory, waypoints, True).calculate()
    plot_all(trajectory, waypoints, metrics=metrics)

    assert len(plt.get_fignums()) == 1
    assert show_calls == [True]
    figure = plt.gcf()
    titles = [axis.get_title() for axis in figure.axes]
    left_titles = [axis.get_title(loc="left") for axis in figure.axes]
    assert "Environment and Path Tracking" in titles
    assert "Sensor vs. Kalman Estimate" in titles
    assert "Velocity Tracking" in titles
    assert "Control Effort" in titles
    assert "Distance to Current Waypoint" in titles
    assert "Guidance Progress" in titles
    assert "Mission Metrics" in left_titles
    plt.close("all")


def test_metrics_can_use_a_window_without_plots(monkeypatch):
    plt.close("all")
    show_calls = []
    monkeypatch.setattr(plt, "show", lambda: show_calls.append(True))
    trajectory = create_trajectory()
    waypoints = [Waypoint(x=2.0, y=0.0)]
    metrics = RunMetrics(trajectory, waypoints, True).calculate()

    plot_metrics(metrics)

    assert len(plt.get_fignums()) == 1
    assert show_calls == [True]
    assert plt.gca().get_title(loc="left") == "Mission Metrics"
    plt.close("all")
