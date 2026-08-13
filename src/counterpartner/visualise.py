"""
Copyright 2026 RICHARD TJÖRNHAMMAR

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_besta_corrections(
    result,
    value_col="Lön",
    group_col="Kön",
    category_col="Besta",
    group1_label="Kvinna",
    group2_label="Man",
    n_categories=None,
    figsize=(10, 6),
    same_y_scale=True,
):
    """
    Plot salary corrections separately for each BESTA category.

    Parameters
    ----------
    result : pandas.DataFrame
        Output from SymmetricOptimalTransportFairness.fit_transform()
        or iterate().

    value_col : str
        Original salary column.

    group_col : str
        Column identifying the two groups.

    category_col : str
        BESTA/category column.

    group1_label : str
        Label for first group.

    group2_label : str
        Label for second group.

    n_categories : int
        Number of BESTA categories to plot.

    figsize : tuple
        Figure size.

    same_y_scale : bool
        If True, all figures use the same y-axis limits.
    """

    df = result.copy()
    if n_categories is None :
        n_categories = len(df.groupby(category_col).apply(len).index.values)
    # ------------------------------------------------------------
    # Make sure the quality is treated consistently
    # ------------------------------------------------------------

    df["_quality"] = (
        df[category_col]
        .astype(str)
    )

    # ------------------------------------------------------------
    # Select categories
    # ------------------------------------------------------------

    categories = (
        df["_quality"]
        .drop_duplicates()
        .tolist()
    )

    categories = categories[
        :n_categories
    ]

    # ------------------------------------------------------------
    # Calculate common y-axis limits
    # ------------------------------------------------------------

    if same_y_scale:

        max_abs_adjustment = (
            np.abs(
                df["adjustment"]
            )
            .max()
        )

        # Avoid zero-width axis
        if (
            not np.isfinite(
                max_abs_adjustment
            )
            or
            max_abs_adjustment == 0
        ):
            max_abs_adjustment = 1.0

        y_min = -0.05 * max_abs_adjustment
        y_max = 1.05 * max_abs_adjustment

    # ------------------------------------------------------------
    # Plot each quality category
    # ------------------------------------------------------------

    figures = []

    for quality in categories:

        sub = df[
            df["_quality"]
            ==
            quality
        ].copy()

        group1 = sub[
            sub[group_col]
            ==
            group1_label
        ].copy()

        group2 = sub[
            sub[group_col]
            ==
            group2_label
        ].copy()

        # --------------------------------------------------------
        # Sort by original salary
        # --------------------------------------------------------

        group1 = group1.sort_values(
            value_col
        )

        group2 = group2.sort_values(
            value_col
        )

        # Individual rank within group
        group1["rank"] = np.arange(
            1,
            len(group1) + 1
        )

        group2["rank"] = np.arange(
            1,
            len(group2) + 1
        )

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=figsize
        )

        # --------------------------------------------------------
        # Plot corrections
        # --------------------------------------------------------

        if len(group1) > 0:

            ax.scatter(
                group1["rank"],
                group1["adjustment"],
                label=group1_label,
                alpha=0.75,
                s=45
            )

        if len(group2) > 0:

            ax.scatter(
                group2["rank"],
                group2["adjustment"],
                label=group2_label,
                alpha=0.75,
                s=45
            )

        # --------------------------------------------------------
        # Zero correction line
        # --------------------------------------------------------

        ax.axhline(
            0,
            linestyle="--",
            linewidth=1
        )

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        ax.set_title(
            f"{category_col} : {quality}"
        )

        ax.set_xlabel(
            "Rank within group by value"
        )

        ax.set_ylabel(
            "Value correction"
        )

        ax.legend()

        ax.grid(
            alpha=0.25
        )

        # --------------------------------------------------------
        # Consistent y-axis
        # --------------------------------------------------------

        if same_y_scale:

            ax.set_ylim(
                y_min,
                y_max
            )

        # --------------------------------------------------------
        # Improve layout
        # --------------------------------------------------------

        fig.tight_layout()

        figures.append(
            fig
        )

    return figures


def plot_quality_histogram(
    result,
    show_col="adjustment",
    value_col="Lön",
    group_col="Kön",
    category_col="Besta",
    group1_label="Kvinna",
    group2_label="Man",
    n_categories=None,
    figsize=(10, 6),
    same_y_scale=False,
    bins=15,
    x_label="Value correction",
    y_label="Number of individuals"
):
    """
    Plot salary-correction histograms separately for each
    BESTA category.

    Parameters
    ----------
    result : pandas.DataFrame
        Output from SymmetricOptimalTransportFairness.fit_transform()
        or iterate().

    value_col : str
        Original salary/value column.

    group_col : str
        Column identifying the two groups.

    category_col : str
        BESTA/category column.

    group1_label : str
        Label for first group.

    group2_label : str
        Label for second group.

    n_categories : int or None
        Number of BESTA categories to plot.
        None = all categories.

    figsize : tuple
        Figure size.

    same_y_scale : bool
        If True, all histograms use the same frequency
        y-axis scale.

    bins : int
        Number of histogram bins.
    """

    df = result.copy()

    # ------------------------------------------------------------
    # Make sure category is treated consistently
    # ------------------------------------------------------------

    df["_quality"] = (
        df[category_col]
        .astype(str)
    )

    # ------------------------------------------------------------
    # Select categories
    # ------------------------------------------------------------

    categories = (
        df["_quality"]
        .drop_duplicates()
        .tolist()
    )

    if n_categories is not None:
        categories = categories[
            :n_categories
        ]

    # ------------------------------------------------------------
    # Determine common histogram range
    #
    # Using the global range makes histograms comparable between
    # BESTA categories.
    # ------------------------------------------------------------

    all_adjustments = (
        df[show_col]
        .dropna()
        .to_numpy(
            dtype=float
        )
    )

    if len(all_adjustments) == 0:
        return []

    global_min = np.min(
        all_adjustments
    )

    global_max = np.max(
        all_adjustments
    )

    # Handle case where all adjustments are identical
    if global_min == global_max:

        padding = (
            abs(global_min) * 0.05
            if global_min != 0
            else 1.0
        )

        global_min -= padding
        global_max += padding

    # ------------------------------------------------------------
    # Common bins
    # ------------------------------------------------------------

    bin_edges = np.linspace(
        global_min,
        global_max,
        bins + 1
    )

    # ------------------------------------------------------------
    # First determine maximum histogram frequency
    # ------------------------------------------------------------

    max_count = 0

    if same_y_scale:

        for quality in categories:

            sub = df[
                df["_quality"]
                ==
                quality
            ]

            group1 = sub[
                sub[group_col]
                ==
                group1_label
            ][show_col].dropna()

            group2 = sub[
                sub[group_col]
                ==
                group2_label
            ][show_col].dropna()

            count1, _ = np.histogram(
                group1,
                bins=bin_edges
            )

            count2, _ = np.histogram(
                group2,
                bins=bin_edges
            )

            max_count = max(
                max_count,
                count1.max(
                    initial=0
                ),
                count2.max(
                    initial=0
                )
            )

        y_max = (
            max_count * 1.10
            if max_count > 0
            else 1
        )

    # ------------------------------------------------------------
    # Create plots
    # ------------------------------------------------------------

    figures = []

    for quality in categories:

        sub = df[
            df["_quality"]
            ==
            quality
        ].copy()

        group1 = sub[
            sub[group_col]
            ==
            group1_label
        ][show_col].dropna()

        group2 = sub[
            sub[group_col]
            ==
            group2_label
        ][show_col].dropna()

        # --------------------------------------------------------
        # Create figure
        # --------------------------------------------------------

        fig, ax = plt.subplots(
            figsize=figsize
        )

        # --------------------------------------------------------
        # Plot group 1
        # --------------------------------------------------------

        if len(group1) > 0:

            ax.hist(
                group1,
                bins=bin_edges,
                alpha=0.60,
                label=group1_label,
                edgecolor="black",
                linewidth=0.5,
            )

        # --------------------------------------------------------
        # Plot group 2
        # --------------------------------------------------------

        if len(group2) > 0:

            ax.hist(
                group2,
                bins=bin_edges,
                alpha=0.60,
                label=group2_label,
                edgecolor="black",
                linewidth=0.5,
            )

        # --------------------------------------------------------
        # Zero correction line
        # --------------------------------------------------------

        #ax.axvline(
        #    0,
        #    linestyle="--",
        #    linewidth=1,
        #)

        # --------------------------------------------------------
        # Labels
        # --------------------------------------------------------

        ax.set_title(
            f"{category_col} : {quality}"
        )

        ax.set_xlabel(
            x_label
        )

        ax.set_ylabel(
            y_label
        )

        ax.legend()

        ax.grid(
            alpha=0.25,
            axis="y"
        )

        # --------------------------------------------------------
        # Consistent y-axis
        # --------------------------------------------------------

        if same_y_scale:

            ax.set_ylim(
                0,
                y_max
            )

        # --------------------------------------------------------
        # Improve layout
        # --------------------------------------------------------

        fig.tight_layout()

        figures.append(
            fig
        )

    return figures
