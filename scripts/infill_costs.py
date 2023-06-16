# -*- coding: utf-8 -*-
"""Script to infill cost matrices produced by OTP4GB-py."""

##### IMPORTS #####
from __future__ import annotations

import argparse
import datetime
import enum
import functools
import pathlib
import re
import sys
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import pandas as pd
import pydantic
import tqdm
from matplotlib import figure
from matplotlib import pyplot as plt
from matplotlib.backends import backend_pdf
from numpy import polynomial
from pydantic import dataclasses, types
from scipy import stats, optimize

sys.path.extend((".", ".."))
from otp4gb import centroids, config, cost, logging, parameters

##### CONSTANTS #####
LOG = logging.get_logger(__name__)


##### CLASSES #####
@dataclasses.dataclass
class InfillArgs:
    folder: types.DirectoryPath
    infill_columns: tuple[str, ...] = ("mean_duration", "mean_travel_distance")

    @classmethod
    def parse(cls) -> InfillArgs:
        parser = argparse.ArgumentParser(
            description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "folder",
            type=pathlib.Path,
            help="folder containing OTP4GB-py outputs to be infilled",
        )
        parser.add_argument(
            "-c",
            "--infill_columns",
            nargs="*",
            default=cls.infill_columns,
            help="name(s) of columns to infill",
        )

        args = parser.parse_args()
        return InfillArgs(folder=args.folder, infill_columns=args.infill_columns)


class PlotType(enum.Enum):
    HEXBIN = enum.auto()
    SCATTER = enum.auto()


class _Config:
    arbitrary_types_allowed = True


@dataclasses.dataclass(config=_Config)
class PlotData:
    x: pd.Series
    y: pd.Series
    title: str | None = None

    @pydantic.validator("y")
    def _check_index(cls, value: pd.Series, values) -> pd.Series:
        if not value.index.equals(values["x"].index):
            raise ValueError("x and y indices are different")

        return value

    def filter(
        self, min_x: float, max_x: float, min_y: float, max_y: float
    ) -> PlotData:
        x_filter = (self.x > min_x) & (self.x < max_x)
        y_filter = (self.y > min_y) & (self.y < max_y)

        index = self.x.index[x_filter & y_filter]

        return PlotData(x=self.x.loc[index], y=self.y.loc[index], title=self.title)


class AxisLimit(NamedTuple):
    min_x: int | float | None = None
    max_x: int | float | None = None
    min_y: int | float | None = None
    max_y: int | float | None = None

    def infill(self, data: PlotData) -> AxisLimit:
        filtered = data.filter(
            min_x=-np.inf if self.min_x is None else self.min_x,
            max_x=np.inf if self.max_x is None else self.max_x,
            min_y=-np.inf if self.min_y is None else self.min_y,
            max_y=np.inf if self.max_y is None else self.max_y,
        )

        values = []

        for name, value in self._asdict().items():
            if value is not None:
                values.append(value)
                continue

            func = np.min if name.startswith("min") else np.max
            data = filtered.x if name.endswith("x") else filtered.y

            values.append(func(data))

        return AxisLimit(*values)


class InfillMethod(enum.Enum):
    MEAN_RATIO = enum.auto()
    LINEAR = enum.auto()
    POLYNOMIAL_2 = enum.auto()
    POLYNOMIAL_3 = enum.auto()
    POLYNOMIAL_4 = enum.auto()
    EXPONENTIAL = enum.auto()
    LOGARITHMIC = enum.auto()


@dataclasses.dataclass
class InfillFunction:
    function: Callable[[np.ndarray], np.ndarray]
    label: str


##### FUNCTIONS #####
def calculate_crow_fly(
    origins_path: pathlib.Path,
    destinations_path: pathlib.Path | None,
    extents: centroids.Bounds | None = None,
) -> pd.Series:
    if destinations_path is None:
        LOG.info("Calculating crow-fly distances for centroids '%s'", origins_path.name)
    else:
        LOG.info(
            "Calculating crow-fly distance for origins '%s' and destinations '%s'",
            origins_path.name,
            destinations_path.name,
        )

    centroid_data = centroids.load_centroids(
        origins_path,
        destinations_path,
        zone_columns=centroids.ZoneCentroidColumns(),
        extents=extents,
    )
    if centroid_data.destinations is None:
        destinations = centroid_data.origins
    else:
        destinations = centroid_data.destinations

    return parameters.calculate_distance_matrix(
        centroid_data.origins, destinations, parameters.CROWFLY_DISTANCE_CRS
    )


def filter_responses(responses_path: pathlib.Path, max_count: int = 10) -> pathlib.Path:
    output_path = responses_path.with_name(responses_path.stem + "-filtered.jsonl")
    count = 0

    with open(responses_path, "rt", encoding="utf-8") as file:
        with open(output_path, "wt", encoding="utf-8") as out_file:
            for result in tqdm.tqdm(
                cost.iterate_responses(file), desc="Iterating responses"
            ):
                if result.plan is None or len(result.plan.itineraries) == 0:
                    continue

                out_file.write(result.json() + "\n")
                count += 1
                if count > max_count:
                    break

    LOG.info("Written filtered responses: %s", output_path.name)
    return output_path


def plot_axes(
    ax: plt.Axes,
    data: PlotData,
    plot_type: PlotType,
    axis_limit: AxisLimit,
) -> None:
    if plot_type == PlotType.HEXBIN:
        hb = ax.hexbin(data.x, data.y, extent=axis_limit, mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count", aspect=40)

    elif plot_type == PlotType.SCATTER:
        ax.scatter(data.x, data.y, rasterized=len(data.x) > 1000)
        ax.set_ylim(axis_limit.min_y, axis_limit.max_y)
        ax.set_xlim(axis_limit.min_x, axis_limit.max_x)

    ax.annotate(
        f"Total Count\n{len(data.x):,}",
        (0.8, 0.05),
        xycoords="axes fraction",
        bbox=dict(boxstyle="round", facecolor="white"),
    )

    ax.set_xlabel(data.x.name)
    ax.set_ylabel(data.y.name)

    if data.title is not None:
        ax.set_title(data.title)


def plot(
    *plot_data: PlotData,
    title: str,
    plot_type: PlotType,
    axis_limit: AxisLimit,
    fit_function: InfillFunction | None = None,
    subplots_kwargs: dict[str, Any] = dict(),
) -> figure.Figure:
    default_figure_kwargs = dict(figsize=(15, 10), constrained_layout=True)
    default_figure_kwargs.update(subplots_kwargs)

    ncols = len(plot_data) if len(plot_data) > 0 else None
    fig, axes = plt.subplots(ncols=ncols, **default_figure_kwargs)
    fig.suptitle(title)

    for ax, data in zip(axes, plot_data):
        limit = axis_limit.infill(data)
        plot_axes(ax, data, plot_type, limit)

        if fit_function is not None:
            x = np.arange(limit.min_x, limit.max_x, 1)
            ax.plot(x, fit_function.function(x), "--", c="C1", label=fit_function.label)
            ax.legend()

    return fig


def _infill_method_ratio(x: pd.Series, y: pd.Series) -> InfillFunction:
    ratio = np.mean(y / x)

    return InfillFunction(
        function=lambda arr: arr * ratio, label=f"Mean Ratio\ny={ratio:.2f}x"
    )


def _infill_method_linear(x: pd.Series, y: pd.Series) -> InfillFunction:
    result = stats.linregress(x, y)
    return InfillFunction(
        function=lambda arr: (arr * result.slope) + result.intercept,
        label=f"Linear Regression\ny={result.slope:.2e}x + {result.intercept:.2e}",
    )


def _infill_method_polynomial(
    x: pd.Series, y: pd.Series, degree: int
) -> InfillFunction:
    poly = polynomial.Polynomial.fit(x, y, degree)

    label = f"Polynomial degree {degree}\n$y = "
    for i, value in enumerate(reversed(poly.coef)):
        power = degree - i
        label += f"+ {value:.1e}"

        if power > 1:
            label += f"x^{power}"
        elif power == 1:
            label += "x"

    label += "$"

    return InfillFunction(function=poly, label=label)


def infill_metric(
    metric: pd.Series,
    distances: pd.Series,
    plot_file: pathlib.Path,
    method: InfillMethod,
) -> pd.Series:
    metric = metric.dropna()
    metric.name = re.sub(r"[\s_]+", " ", metric.name).title()

    # TODO Figure out issue with zones in metric which aren't in distances
    data = pd.concat([metric, distances], axis=1)

    for column, values in data.items():
        nan_values = values.isna().sum()
        if nan_values > 0:
            LOG.warning("%s Nan values in %s column", f"{nan_values:,}", column)

    before = len(data)
    data = data.dropna(how="any")
    after = len(data)
    message = (
        f"{after:,} ({after / before:.0%}) rows remaining "
        f"after dropping missing values, {before:,} before"
    )
    LOG.info(message)

    plot_data = [
        PlotData(
            x=data[distances.name],
            y=data[metric.name],
            title=f"Before Infilling\n{metric.name} vs {distances.name}",
        )
    ]

    missing = distances.index[~distances.index.isin(data.index)]
    LOG.info("Infilling %s values with %s method", f"{len(missing):,}", method)

    infill_methods = {
        InfillMethod.MEAN_RATIO: _infill_method_ratio,
        InfillMethod.LINEAR: _infill_method_linear,
        InfillMethod.POLYNOMIAL_2: functools.partial(
            _infill_method_polynomial, degree=2
        ),
        InfillMethod.POLYNOMIAL_3: functools.partial(
            _infill_method_polynomial, degree=3
        ),
        InfillMethod.POLYNOMIAL_4: functools.partial(
            _infill_method_polynomial, degree=4
        ),
    }

    if method not in infill_methods:
        raise ValueError(f"unknown infilling method {method}")

    infill_function = infill_methods[method](data[distances.name], data[metric.name])
    calculated = infill_function.function(distances.loc[missing])

    if calculated.index.isin(metric.index).sum() > 0:
        raise ValueError("Oops recalculated existing metrics")

    infilled = pd.concat([metric, calculated], axis=0)
    infilled.name = "Infilled " + metric.name

    infilled_data = pd.concat([distances, infilled], axis=1)

    plot_data.append(
        PlotData(
            x=infilled_data.loc[missing, distances.name],
            y=infilled_data.loc[missing, infilled.name],
            title=f"Only {infilled.name} vs {distances.name}",
        )
    )
    plot_data.append(
        PlotData(
            x=infilled_data[distances.name],
            y=infilled_data[infilled.name],
            title=f"After Infilling\n{infilled.name} vs {distances.name}",
        )
    )

    axis_limit = AxisLimit(min_x=0, max_x=200, min_y=0, max_y=None)
    infilled_limits = (axis_limit.infill(i) for i in plot_data)
    max_y = functools.reduce(max, (i.max_y for i in infilled_limits), 0)
    axis_limit = AxisLimit(axis_limit.min_x, axis_limit.max_x, axis_limit.min_y, max_y)

    with backend_pdf.PdfPages(plot_file) as pdf:
        for pt in PlotType:
            fig = plot(
                *plot_data,
                title=f"Infilling Comparison for {plot_file.stem} - {metric.name}",
                plot_type=pt,
                axis_limit=axis_limit,
                fit_function=infill_function,
                subplots_kwargs=dict(sharey=True, figsize=(20, 8)),
            )

            pdf.savefig(fig)
            plt.close(fig)

    LOG.info("Saved plots to %s", plot_file.name)

    return infilled


def infill_costs(
    metrics_path: pathlib.Path,
    columns: Sequence[str],
    distances: pd.Series,
    infill_folder: pathlib.Path,
) -> None:
    LOG.info("Reading '%s'", metrics_path)
    metrics = pd.read_csv(metrics_path, index_col=["origin", "destination"])

    if distances.isna().sum() > 0:
        raise ValueError(f"distances has {distances.isna().sum()} Nan values")

    for method in InfillMethod:
        plot_folder = infill_folder / f"plots - {method.name.lower()}"
        plot_folder.mkdir(exist_ok=True)
        infilled_metrics = []

        for column in columns:
            if column not in metrics.columns:
                LOG.error(
                    "Metric column '%s' not found in '%s'", column, metrics_path.name
                )
                continue

            infilled_metrics.append(
                infill_metric(
                    metrics[column],
                    distances,
                    plot_folder / (metrics_path.stem + f"-{column}.pdf"),
                    method,
                )
            )

        infilled_df = pd.concat(infilled_metrics, axis=1)

        out_path = infill_folder / (
            metrics_path.stem + f"-infilled_{method.name.lower()}.csv"
        )
        infilled_df.to_csv(out_path)
        LOG.info("Written: %s", out_path)


def main(params: config.ProcessConfig, arguments: InfillArgs) -> None:
    logging.initialise_logger("", arguments.folder / "logs/infill_costs.log")

    origin_path = config.ASSET_DIR / params.centroids
    destination_path = None
    if params.destination_centroids is not None:
        destination_path = config.ASSET_DIR / params.destination_centroids

    distances = calculate_crow_fly(origin_path, destination_path, params.extents)
    distances = distances / 1000
    distances.name = "Crow-Fly Distance (km)"

    for time_period in params.time_periods:
        travel_datetime = datetime.datetime.combine(
            params.date, time_period.travel_time
        )
        # Assume time is in local timezone
        travel_datetime = travel_datetime.astimezone()
        LOG.info(
            "Given date / time is assumed to be in local timezone: %s",
            travel_datetime.tzinfo,
        )

        for modes in params.modes:
            matrix_path = arguments.folder / (
                f"costs/{time_period.name}/"
                f"{'_'.join(modes)}_costs_{travel_datetime:%Y%m%dT%H%M}-metrics.csv"
            )

            if not matrix_path.is_file():
                raise FileNotFoundError(matrix_path)

            recalculated_path = matrix_path.with_name(
                matrix_path.stem + "-recalculated.csv"
            )
            recalculated_metrics_path = recalculated_path.with_name(
                recalculated_path.stem + "-metrics.csv"
            )
            if not recalculated_metrics_path.is_file():
                responses_path = matrix_path.with_name(
                    matrix_path.stem.replace("-metrics", ".csv-response_data.jsonl")
                )
                cost.cost_matrix_from_responses(
                    responses_path,
                    recalculated_path,
                    params.iterinary_aggregation_method,
                )

            infill_folder = recalculated_metrics_path.parent / "infilled"
            infill_folder.mkdir(exist_ok=True)
            infill_costs(
                recalculated_metrics_path,
                arguments.infill_columns,
                distances,
                infill_folder,
            )


def _run() -> None:
    args = InfillArgs.parse()

    params = config.load_config(args.folder)
    main(params, args)


if __name__ == "__main__":
    _run()
