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
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.optimize import linear_sum_assignment

def fractional_rank_match_hungarian(group_a, group_b):
    """
    Match elements between group_a and group_b using fractional ranks and Hungarian algorithm.
    Returns one-to-one matches between the two groups (no replacement).

    Parameters:
    group_a, group_b: array-like, 1D sequences of numeric values.

    Returns:
    matched_pairs: list of tuples [(idx_a, idx_b, val_a, val_b), ...]
    matches_a_to_b: dict mapping {index_in_a: index_in_b}
    matches_b_to_a: dict mapping {index_in_b: index_in_a} (or None if unmatched)
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    # Step 1: compute fractional ranks in pooled sample
    combined = np.concatenate([group_a, group_b])
    ranks = rankdata(combined, method='average')
    n_total = len(combined)
    frac_ranks = (ranks - 0.5) / n_total

    # Split back
    n_a = len(group_a)
    n_b = len(group_b)
    frac_a = frac_ranks[:n_a]
    frac_b = frac_ranks[n_a:]

    # Step 2: create cost matrix (absolute differences in fractional ranks)
    # Size: n_a x n_b
    cost_matrix = np.abs(frac_a[:, np.newaxis] - frac_b[np.newaxis, :])

    # Step 3: apply Hungarian algorithm
    # If groups differ in size, we need a square matrix with dummy nodes
    n_nodes = max(n_a, n_b)

    # Create square cost matrix padded with large values (or 0 for dummy matches)
    square_cost = np.full((n_nodes, n_nodes), np.max(cost_matrix) * 2)
    square_cost[:n_a, :n_b] = cost_matrix

    # For dummy rows/columns, cost is high to discourage matching if possible
    # But we need to allow them. Set dummy costs to a moderate value.
    # Better: use a large constant so real matches are preferred
    dummy_cost = np.max(cost_matrix) * 2
    square_cost[n_a:, :n_b] = dummy_cost # dummy rows
    square_cost[:n_a, n_b:] = dummy_cost # dummy columns
    square_cost[n_a:, n_b:] = 0 # dummy-dummy matches cost 0

    # Apply Hungarian
    row_indices, col_indices = linear_sum_assignment(square_cost)

    # Step 4: extract only real matches (non-dummy)
    matched_pairs = []
    matches_a_to_b = {}
    matches_b_to_a = {}

    for r, c in zip(row_indices, col_indices):
        if r < n_a and c < n_b: # both are real elements
            matched_pairs.append({
                'index_a': r,
                'index_b': c,
                'value_a': group_a[r],
                'value_b': group_b[c],
                'frac_rank_a': frac_a[r],
                'frac_rank_b': frac_b[c],
                'cost': cost_matrix[r, c]
            })
            matches_a_to_b[r] = c
            matches_b_to_a[c] = r

    # For unmatched elements (optional)
    unmatched_a = [i for i in range(n_a) if i not in matches_a_to_b]
    unmatched_b = [i for i in range(n_b) if i not in matches_b_to_a]

    return {
        'matched_pairs': matched_pairs,
        'matches_a_to_b': matches_a_to_b,
        'matches_b_to_a': matches_b_to_a,
        'unmatched_a': unmatched_a,
        'unmatched_b': unmatched_b,
        'n_matches': len(matched_pairs)
    }



def fractional_rank_match_hungarian_simple(group_a, group_b):
    """
    Returns matched pairs as parallel arrays.
    """
    result = fractional_rank_match_hungarian(group_a, group_b)

    if not result['matched_pairs']:
        return np.array([]), np.array([]), np.array([]), np.array([])

    idx_a = np.array([p['index_a'] for p in result['matched_pairs']])
    idx_b = np.array([p['index_b'] for p in result['matched_pairs']])
    val_a = np.array([p['value_a'] for p in result['matched_pairs']])
    val_b = np.array([p['value_b'] for p in result['matched_pairs']])

    return idx_a, idx_b, val_a, val_b


def fractional_rank_match_with_replacement(group_a, group_b):
    """
    Match ALL elements from both groups to their closest fractional rank in the OTHER group.
    Returns bidirectional matches (many-to-one allowed, with replacement).

    Parameters:
    group_a, group_b: array-like, 1D sequences of numeric values.

    Returns:
    matches_a_to_b: dict mapping {index_in_a: index_in_b} (each a matched to closest b)
    matches_b_to_a: dict mapping {index_in_b: index_in_a} (each b matched to closest a)
    """
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    # Step 1: compute fractional ranks in pooled sample
    combined = np.concatenate([group_a, group_b])
    ranks = rankdata(combined, method='average')
    n_total = len(combined)
    frac_ranks = (ranks - 0.5) / n_total

    # Split back
    n_a = len(group_a)
    frac_a = frac_ranks[:n_a]
    frac_b = frac_ranks[n_a:]

    # Step 2: match each element in A to closest in B
    matches_a_to_b = {}
    for i, f_a in enumerate(frac_a):
        idx_b = np.argmin(np.abs(frac_b - f_a))
        matches_a_to_b[i] = idx_b

    # Step 3: match each element in B to closest in A
    matches_b_to_a = {}
    for j, f_b in enumerate(frac_b):
        idx_a = np.argmin(np.abs(frac_a - f_b))
        matches_b_to_a[j] = idx_a

    return {
        'matches_a_to_b': matches_a_to_b,
        'matches_b_to_a': matches_b_to_a,
        'frac_ranks_a': frac_a,
        'frac_ranks_b': frac_b
    }


def get_matched_pairs(group_a, group_b):
    """
    Returns all matched pairs as a list, clearly showing which matches which.
    """
    result = fractional_rank_match_with_replacement(group_a, group_b)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    pairs = []

    # Pairs from A perspective
    for idx_a, idx_b in result['matches_a_to_b'].items():
        pairs.append({
            'from_group': 'A',
            'index_a': idx_a,
            'index_b': idx_b,
            'value_a': group_a[idx_a],
            'value_b': group_b[idx_b],
            'frac_a': result['frac_ranks_a'][idx_a],
            'frac_b': result['frac_ranks_b'][idx_b]
        })

    return pairs


def get_compact_matches(group_a, group_b):
    """
    Returns compact representation: arrays of matched indices and values.
    Note: Each B element may appear multiple times if A is larger.
    """
    result = fractional_rank_match_with_replacement(group_a, group_b)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)

    # A -> B matches
    idx_a_list = list(result['matches_a_to_b'].keys())
    idx_b_list = [result['matches_a_to_b'][i] for i in idx_a_list]
    val_a_list = [group_a[i] for i in idx_a_list]
    val_b_list = [group_b[j] for j in idx_b_list]

    # B -> A matches
    idx_b_rev = list(result['matches_b_to_a'].keys())
    idx_a_rev = [result['matches_b_to_a'][j] for j in idx_b_rev]
    val_b_rev = [group_b[j] for j in idx_b_rev]
    val_a_rev = [group_a[i] for i in idx_a_rev]

    return {
        'a_to_b_indices': (np.array(idx_a_list), np.array(idx_b_list)),
        'a_to_b_values': (np.array(val_a_list), np.array(val_b_list)),
        'b_to_a_indices': (np.array(idx_b_rev), np.array(idx_a_rev)),
        'b_to_a_values': (np.array(val_b_rev), np.array(val_a_rev))
    }


class NaiveValueEquityAdjuster:
    """
    Fair value adjustment model using within-category quantile matching.

    Main principles:
    ----------------
    1. Compare only comparable categories
    2. No value reductions
    3. Within-group percentile fairness
    4. Small group robustness
    """

    def __init__(
        self,
        value_col           = "Lön",
        group_col           = "Kön",
        category_col        = "Besta",
        instance1_label     = None, #"K",
        instance2_label     = None, #"M",
        min_group_size      = 3,
        shrinkage_strength  = 0.5,
    ):
        if instance1_label is None or instance2_label is None :
            print("ERROR:NaiveValueEquityAdjuster")
            print("ERROR:Please specify instance1_label and instance2_label")
            print("ERROR:Corresponding to the wanted group_col comparison")
            exit(1)
        self.value_col      = value_col
        self.group_col      = group_col
        self.category_col   = category_col

        self.instance1_label    = instance1_label
        self.instance2_label    = instance2_label

        self.min_group_size     = min_group_size
        self.shrinkage_strength = shrinkage_strength

    def _quantile_match(self, source, target):
        """
        Match source salaries to target distribution
        using percentile interpolation.
        """

        source = np.asarray(source)
        target = np.asarray(target)

        if len(target) == 0:
            return source.copy()

        source_rank = rankdata(source, method="average")
        q = (source_rank - 0.5) / len(source)

        target_sorted = np.sort(target)
        grid = np.linspace(0, 1, len(target_sorted))

        matched = np.interp(
            q,
            grid,
            target_sorted,
            left=target_sorted[0],
            right=target_sorted[-1],
        )

        return matched

    def _shrinkage_reference(
        self,
        local_reference,
        global_reference,
        n_group
    ):
        """
        Shrink small groups toward global estimate.
        """

        alpha = n_group / (
            n_group + self.shrinkage_strength
        )

        return (
            alpha * local_reference
            + (1 - alpha) * global_reference
        )

    def fit_transform(self, df):
        """
        Compute fair salary adjustments.
        """

        df = df.copy()

        results = []

        global_instance2 = df.loc[
            df[self.group_col] == self.instance2_label,
            self.value_col
        ].values

        global_instance1 = df.loc[
            df[self.group_col] == self.instance1_label,
            self.value_col
        ].values

        all_categories = (
            df[self.category_col]
            .astype(str)
            .unique()
        )

        for cat in all_categories:

            sub = df[
                df[self.category_col]
                .astype(str)
                == str(cat)
            ].copy()

            instance2 = sub[
                sub[self.group_col]
                == self.instance2_label
            ]

            instance1 = sub[
                sub[self.group_col]
                == self.instance1_label
            ]

            if (
                len(instance2) < self.min_group_size
                or len(instance1)
                < self.min_group_size
            ):
                continue

            instance2_value = instance2[
                self.value_col
            ].values

            instance1_value = instance1[
                self.value_col
            ].values

            # instance1 → instance2 reference
            local_reference = (
                self._quantile_match(
                    instance1_value,
                    instance2_value
                )
            )

            # global fallback
            global_reference = (
                self._quantile_match(
                    instance1_value,
                    global_instance2
                )
            )

            reference = (
                self._shrinkage_reference(
                    local_reference,
                    global_reference,
                    len(sub)
                )
            )

            adjustment = np.maximum(
                reference - instance1_value,
                0
            )

            instance1_result = instance1.copy()

            instance1_result[
                "reference_value"
            ] = reference

            instance1_result[
                "adjustment"
            ] = adjustment

            instance1_result[
                "new_salary"
            ] = (
                instance1_result[self.value_col]
                + adjustment
            )

            instance1_result[
                "Category_group"
            ] = cat

            results.append(instance1_result)

        if len(results) == 0:
            return pd.DataFrame()

        report = pd.concat(results)

        return report.sort_values(
            "adjustment",
            ascending=False
        )


class OptimalTransportFairness:
    """
    Fair value adjustment using constrained
    optimal transport.

    Principles
    ----------
    1. Compare only within comparable groups e.g.
       (BESTA, grade, category)

    2. Minimize total value adjustment

    3. Positive-only corrections:
       no salary reductions

    4. Quantile optimal transport
       (exact 1D Wasserstein map)
    """

    def __init__(
        self,
        value_col       = "Lön",
        group_col       = "Kön",
        category_col    = "Besta",
        instance1_label = None,
        instance2_label = None,
        min_group_size  = 3,
        regularization  = 0.25,
    ):
        if instance1_label is None or instance2_label is None :
            print("ERROR:OptimalTransportFairness")
            print("ERROR:Please specify instance1_label and instance2_label")
            print("ERROR:Corresponding to the wanted group_col comparison")
            exit(1)
        self.value_col      = value_col
        self.group_col      = group_col
        self.category_col   = category_col

        self.instance1_label    = instance1_label
        self.instance2_label    = instance2_label

        self.min_group_size = min_group_size
        self.regularization = regularization

    # -----------------------------------
    # Quantile OT map
    # -----------------------------------

    def _transport_map(
        self,
        source,
        target
    ):
        """
        Exact 1D optimal transport map.

        Maps source quantiles to target
        quantiles.
        """

        source = np.asarray(source)
        target = np.asarray(target)

        if len(target) == 0:
            return source.copy()

        rank = rankdata(
            source,
            method="average"
        )

        q = (
            rank - 0.5
        ) / len(source)

        target_sorted = np.sort(target)

        target_quantiles = np.linspace(
            0,
            1,
            len(target_sorted)
        )

        transported = np.interp(
            q,
            target_quantiles,
            target_sorted,
            left=target_sorted[0],
            right=target_sorted[-1]
        )

        return transported

    # -----------------------------------
    # Positive fairness projection
    # -----------------------------------

    def _positive_projection(
        self,
        original_value,
        transported_value
    ):
        """
        Positivity constraint:

        No salary reductions allowed.
        """

        return np.maximum(
            original_value,
            transported_value
        )

    # -----------------------------------
    # Regularized adjustment
    # -----------------------------------

    def _regularized_update(
        self,
        current,
        projected
    ):
        """
        Conservative adjustment.

        Avoids extreme jumps.
        """

        return (
            current
            +
            self.regularization
            *
            (
                projected
                - current
            )
        )

    # -----------------------------------
    # Solve fairness problem
    # -----------------------------------

    def fit_transform(
        self,
        df
    ):

        df = df.copy()

        outputs = []

        for cat in (
            df[self.category_col]
            .astype(str)
            .unique()
        ):

            sub = df[
                df[self.category_col]
                .astype(str)
                ==
                str(cat)
            ].copy()

            instance1 = sub[
                sub[self.group_col]
                ==
                self.instance1_label
            ].copy()

            instance2 = sub[
                sub[self.group_col]
                ==
                self.instance2_label
            ].copy()

            if (
                len(instance1)
                <
                self.min_group_size
                or
                len(instance2)
                <
                self.min_group_size
            ):
                continue

            instance1_value = instance1[
                self.value_col
            ].values

            instance2_value = instance2[
                self.value_col
            ].values

            # -------------------------
            # Optimal transport
            # -------------------------

            transported = (
                self._transport_map(
                    instance1_value,
                    instance2_value
                )
            )

            # -------------------------
            # Positivity constraint
            # -------------------------

            projected = (
                self._positive_projection(
                    instance1_value,
                    transported
                )
            )

            # -------------------------
            # Conservative update
            # -------------------------

            fair_value = (
                self._regularized_update(
                    instance1_value,
                    projected
                )
            )

            instance1[
                "transport_salary"
            ] = transported

            instance1[
                "fair_value"
            ] = fair_value

            instance1[
                "adjustment"
            ] = (
                fair_value
                -
                instance1_value
            )

            instance1[
                "Category_group"
            ] = cat

            instance1[
                "relative_gap"
            ] = (
                instance1["adjustment"]
                /
                instance1_value
            )

            outputs.append(instance1)

        if len(outputs) == 0:
            return pd.DataFrame()

        result = pd.concat(outputs)

        return result.sort_values(
            "adjustment",
            ascending=False
        )





class SymmetricOptimalTransportFairness:
    """
    Symmetric fairness adjustment using 1D optimal transport.

    Policy
    ------
    1. Compare only within comparable categories.
    2. Treat the two groups symmetrically.
    3. Never reduce an individual's value.
    4. Eliminate the observed quantile disparity.
    5. Minimize the total required upward adjustment.
    6. Preserve rank ordering within each group.

    Mathematical formulation
    -------------------------
    For two groups with empirical quantile functions

        Q1(p)
        Q2(p)

    define the common target as the upper quantile envelope:

        Q*(p) = max(Q1(p), Q2(p))

    This is the minimal pointwise target satisfying

        Q*(p) >= Q1(p)
        Q*(p) >= Q2(p)

    for every percentile p.

    Each individual is then transported to the corresponding
    percentile of Q*.

    Optional regularization applies only a fraction lambda of
    the required upward correction:

        x_new = x + lambda * (target - x)

    where 0 < lambda <= 1.

    lambda = 1
        Full minimal correction in one step.

    lambda < 1
        Conservative correction applied iteratively.
    """

    def __init__(
        self,
        value_col="Lön",
        group_col="Kön",
        category_col="Besta",
        instance1_label=None,
        instance2_label=None,
        min_group_size=3,
        regularization=1.0,
        quantile_grid_size=201,
    ):
        if instance1_label is None or instance2_label is None:
            raise ValueError(
                "Please specify instance1_label and instance2_label."
            )

        if not 0 < regularization <= 1:
            raise ValueError(
                "regularization must satisfy 0 < regularization <= 1."
            )

        if quantile_grid_size < 2:
            raise ValueError(
                "quantile_grid_size must be >= 2."
            )

        self.value_col = value_col
        self.group_col = group_col
        self.category_col = category_col

        self.instance1_label = instance1_label
        self.instance2_label = instance2_label

        self.min_group_size = min_group_size
        self.regularization = regularization
        self.quantile_grid_size = quantile_grid_size

    # ============================================================
    # Quantile grid
    # ============================================================

    def _quantile_grid(self):
        """
        Common percentile grid used for both groups.
        """

        return np.linspace(
            0.0,
            1.0,
            self.quantile_grid_size
        )

    # ============================================================
    # Empirical quantile function
    # ============================================================

    def _quantile_function(
        self,
        values,
        q_grid
    ):
        """
        Empirical quantile function.

        Q(p) = empirical p-quantile.
        """

        values = np.asarray(
            values,
            dtype=float
        )

        values = values[
            np.isfinite(values)
        ]

        if len(values) == 0:
            raise ValueError(
                "Cannot construct quantile function "
                "from an empty value array."
            )

        return np.quantile(
            values,
            q_grid
        )

    # ============================================================
    # Upper Wasserstein envelope
    # ============================================================

    def _upper_envelope(
        self,
        values1,
        values2,
        q_grid
    ):
        """
        Construct the symmetric minimal positive target.

        Q*(p) = max(Q1(p), Q2(p))

        This is the smallest quantile function that is
        simultaneously greater than or equal to both
        input quantile functions.
        """

        q1 = self._quantile_function(
            values1,
            q_grid
        )

        q2 = self._quantile_function(
            values2,
            q_grid
        )

        target = np.maximum(
            q1,
            q2
        )

        return q1, q2, target

    # ============================================================
    # Individual percentile positions
    # ============================================================

    def _empirical_percentiles(
        self,
        values
    ):
        """
        Calculate the percentile position of every observation.

        q_i = (rank_i - 0.5) / n

        This corresponds to the fractional-rank convention
        used in the original implementation.
        """

        values = np.asarray(
            values,
            dtype=float
        )

        n = len(values)

        if n == 0:
            return np.array([])

        ranks = rankdata(
            values,
            method="average"
        )

        return (
            ranks - 0.5
        ) / n

    # ============================================================
    # Transport to arbitrary quantile function
    # ============================================================

    def _transport_to_target(
        self,
        source,
        target_quantiles,
        q_grid
    ):
        """
        Transport source observations to a target quantile
        function.

        Each observation retains its percentile rank.

        Therefore the transport is monotonic and does not
        reorder individuals within a group.
        """

        source = np.asarray(
            source,
            dtype=float
        )

        q = self._empirical_percentiles(
            source
        )

        transported = np.interp(
            q,
            q_grid,
            target_quantiles,
            left=target_quantiles[0],
            right=target_quantiles[-1]
        )

        return transported

    # ============================================================
    # Positive projection
    # ============================================================

    def _positive_projection(
        self,
        original,
        target
    ):
        """
        Enforce the policy constraint:

            adjusted_value >= original_value

        No salary reductions are permitted.
        """

        return np.maximum(
            original,
            target
        )

    # ============================================================
    # Regularized update
    # ============================================================

    def _regularized_update(
        self,
        current,
        target
    ):
        """
        Apply a fraction of the required correction.

        lambda = 1
            Full correction.

        lambda < 1
            Conservative correction.
        """

        return (
            current
            +
            self.regularization
            *
            (
                target
                -
                current
            )
        )

    # ============================================================
    # Calculate correction for one category
    # ============================================================

    def _solve_category(
        self,
        instance1,
        instance2,
        category
    ):
        """
        Solve the symmetric OT problem for one category.
        """

        values1 = instance1[
            self.value_col
        ].to_numpy(
            dtype=float
        )

        values2 = instance2[
            self.value_col
        ].to_numpy(
            dtype=float
        )

        # --------------------------------------------------------
        # Remove invalid observations
        # --------------------------------------------------------

        valid1 = np.isfinite(values1)
        valid2 = np.isfinite(values2)

        values1 = values1[valid1]
        values2 = values2[valid2]

        if (
            len(values1) < self.min_group_size
            or
            len(values2) < self.min_group_size
        ):
            return None

        # --------------------------------------------------------
        # Quantile grid
        # --------------------------------------------------------

        q_grid = self._quantile_grid()

        # --------------------------------------------------------
        # Quantile distributions
        # --------------------------------------------------------

        q1, q2, target = (
            self._upper_envelope(
                values1,
                values2,
                q_grid
            )
        )

        # --------------------------------------------------------
        # Transport both groups to common target
        # --------------------------------------------------------

        transported1 = (
            self._transport_to_target(
                values1,
                target,
                q_grid
            )
        )

        transported2 = (
            self._transport_to_target(
                values2,
                target,
                q_grid
            )
        )

        # --------------------------------------------------------
        # Positive-only projection
        # --------------------------------------------------------

        projected1 = (
            self._positive_projection(
                values1,
                transported1
            )
        )

        projected2 = (
            self._positive_projection(
                values2,
                transported2
            )
        )

        # --------------------------------------------------------
        # Regularized update
        # --------------------------------------------------------

        fair1 = (
            self._regularized_update(
                values1,
                projected1
            )
        )

        fair2 = (
            self._regularized_update(
                values2,
                projected2
            )
        )

        # --------------------------------------------------------
        # Build output
        # --------------------------------------------------------

        result1 = instance1.loc[
            valid1
        ].copy()

        result2 = instance2.loc[
            valid2
        ].copy()

        # --------------------------------------------------------
        # Group 1
        # --------------------------------------------------------

        result1[
            "transport_value"
        ] = transported1

        result1[
            "fair_value"
        ] = fair1

        result1[
            "adjustment"
        ] = (
            fair1
            -
            values1
        )

        result1[
            "relative_gap"
        ] = np.divide(
            result1["adjustment"].to_numpy(),
            values1,
            out=np.zeros_like(values1),
            where=values1 != 0
        )

        result1[
            "fairness_group"
        ] = self.instance1_label

        # --------------------------------------------------------
        # Group 2
        # --------------------------------------------------------

        result2[
            "transport_value"
        ] = transported2

        result2[
            "fair_value"
        ] = fair2

        result2[
            "adjustment"
        ] = (
            fair2
            -
            values2
        )

        result2[
            "relative_gap"
        ] = np.divide(
            result2["adjustment"].to_numpy(),
            values2,
            out=np.zeros_like(values2),
            where=values2 != 0
        )

        result2[
            "fairness_group"
        ] = self.instance2_label

        # --------------------------------------------------------
        # Metadata
        # --------------------------------------------------------

        result1[
            "Category_group"
        ] = category

        result2[
            "Category_group"
        ] = category

        # --------------------------------------------------------
        # Store category-level information
        # --------------------------------------------------------

        category_info = {
            "category": category,
            "q_grid": q_grid,
            "q1": q1,
            "q2": q2,
            "target": target,
            "wasserstein_gap": np.trapezoid(
                np.abs(q1 - q2),
                q_grid
            ),
            "total_adjustment_group1": np.sum(
                result1["adjustment"]
            ),
            "total_adjustment_group2": np.sum(
                result2["adjustment"]
            ),
        }

        return (
            result1,
            result2,
            category_info
        )

    def _identity_result(
        self,
        sub,
        status="insufficient_evidence"
    ):
        sub = sub.copy()
        sub["fair_value"]      = sub[self.value_col]
        sub["adjustment"]      = 0.0
        sub["relative_gap"]    = 0.0
        sub["fairness_status"] = status

        return sub

    # ============================================================
    # Full transformation
    # ============================================================

    def fit_transform(
        self,
        df
    ):
        """
        Calculate fairness-adjusted values for all eligible
        categories.

        Returns
        -------
        result : pandas.DataFrame
            Individual-level adjustment results.

        category_info : list
            Quantile-level information for each category.
        """

        outputs = []
        category_information = []

        categories = (
            df[self.category_col]
            .astype(str)
            .unique()
        )

        for category in categories:

            sub = df[
                df[self.category_col]
                .astype(str)
                ==
                str(category)
            ].copy()

            instance1 = sub[
                sub[self.group_col]
                ==
                self.instance1_label
            ].copy()

            instance2 = sub[
                sub[self.group_col]
                ==
                self.instance2_label
            ].copy()

            if (
                len(instance1)
                < self.min_group_size
                or
                len(instance2)
                < self.min_group_size
            ):
                outputs.append(
                    self._identity_result(
                        sub,
                        "insufficient_evidence"
                    )
                )
                continue

            solved = self._solve_category(
                instance1,
                instance2,
                category
            )

            if solved is None:
                outputs.append(
                    self._identity_result(
                        sub,
                        "insufficient_evidence"
                    )
                )
                continue

            result1, result2, info = solved

            outputs.append(result1)
            outputs.append(result2)

            category_information.append(
                info
            )

        if len(outputs) == 0:

            return (
                pd.DataFrame(),
                category_information
            )

        result = pd.concat(
            outputs,
            axis=0
        )

        result = result.sort_values(
            "adjustment",
            ascending=False
        )

        return (
            result,
            category_information
        )

    # ============================================================
    # Iterative correction
    # ============================================================

    def iterate(
        self,
        df,
        tolerance=1e-6,
        max_iterations=100
    ):
        """
        Iteratively apply the minimal positive correction.

        Useful when regularization < 1.

        Returns
        -------
        result : pandas.DataFrame
            Final individual-level values and adjustments.

        history : list
            Convergence information.
        """

        current = df.copy()

        original_values = (
            df[self.value_col]
            .to_numpy(
                dtype=float
            )
        )

        history = []

        for iteration in range(
            max_iterations
        ):

            result, category_info = (
                self.fit_transform(
                    current
                )
            )

            if result.empty:
                return (
                    result,
                    history
                )

            # ----------------------------------------------------
            # Map calculated fair values back to current dataframe
            # ----------------------------------------------------

            updated = current.copy()

            updated.loc[
                result.index,
                self.value_col
            ] = result[
                "fair_value"
            ]

            # ----------------------------------------------------
            # Measure maximum change
            # ----------------------------------------------------

            old_values = (
                current.loc[
                    result.index,
                    self.value_col
                ].to_numpy(
                    dtype=float
                )
            )

            new_values = (
                result[
                    "fair_value"
                ].to_numpy(
                    dtype=float
                )
            )

            max_change = np.max(
                np.abs(
                    new_values
                    -
                    old_values
                )
            )

            total_increase = np.sum(
                new_values
                -
                old_values
            )

            history.append(
                {
                    "iteration": iteration + 1,
                    "max_change": max_change,
                    "total_increment": total_increase,
                }
            )

            current = updated

            if max_change <= tolerance:
                break

        # --------------------------------------------------------
        # Final report
        # --------------------------------------------------------

        final = current.copy()

        final[
            "original_value"
        ] = original_values

        final[
            "fair_value"
        ] = final[
            self.value_col
        ]

        final[
            "adjustment"
        ] = (
            final["fair_value"]
            -
            final["original_value"]
        )

        final[
            "relative_gap"
        ] = np.divide(
            final["adjustment"].to_numpy(),
            final["original_value"].to_numpy(),
            out=np.zeros_like(original_values),
            where=final["original_value"].to_numpy() != 0
        )

        final = final.sort_values(
            "adjustment",
            ascending=False
        )

        return (
            final,
            history
        )


class CatParser:
    """
    Category structure

    XXC.......

    XX = work type
    C  = complexity

    remainder = expert tail
    """

    @staticmethod
    def work_type(code):
        code = str(code)
        return code[:2]

    @staticmethod
    def complexity(code):
        code = str(code)

        if len(code) < 3:
            return None

        return code[2]

    @staticmethod
    def comparison_key(code):

        return (
            CatParser.work_type(code),
            CatParser.complexity(code)
        )


class PooledSymmetricOptimalTransportFairness:

    def __init__(
        self,
        value_col="Lön",
        group_col="Kön",
        category_col="Besta",
        group1_label="Kvinna",
        group2_label="Man",
        tau=10,
        quantile_grid_size=101,
        min_pool_evidence=3,
    ):

        self.value_col = value_col
        self.group_col = group_col
        self.category_col = category_col

        self.group1_label = group1_label
        self.group2_label = group2_label

        self.tau = tau

        self.quantile_grid_size = (
            quantile_grid_size
        )

        self.min_pool_evidence = (
            min_pool_evidence
        )

    # --------------------------------------------------
    # Evidence
    # --------------------------------------------------

    def _evidence(
        self,
        n1,
        n2
    ):
        return min(n1, n2)

    # --------------------------------------------------
    # Quantile function
    # --------------------------------------------------

    def _quantile_function(
        self,
        values,
        q_grid
    ):

        values = np.asarray(values)

        if len(values) == 0:
            return None

        return np.quantile(
            values,
            q_grid,
            method="linear"
        )

    # --------------------------------------------------
    # Transport
    # --------------------------------------------------

    def _transport_to_target(
        self,
        source_values,
        target_quantiles,
        q_grid
    ):

        source_values = np.asarray(
            source_values
        )

        rank = rankdata(
            source_values,
            method="average"
        )

        q = (
            rank - 0.5
        ) / len(source_values)

        transported = np.interp(
            q,
            q_grid,
            target_quantiles
        )

        return transported

    # --------------------------------------------------
    # Candidate pool
    # --------------------------------------------------

    def _pool_candidates(
        self,
        category,
        all_categories
    ):

        key = CatParser.comparison_key(
            category
        )

        output = []

        for c in all_categories:

            if c == category:
                continue

            if (
                CatParser.comparison_key(c)
                ==
                key
            ):
                output.append(c)

        return output

    # --------------------------------------------------
    # Local fairness target
    # --------------------------------------------------

    def _local_target(
        self,
        sub,
        q_grid
    ):

        g1 = sub[
            sub[self.group_col]
            ==
            self.group1_label
        ][self.value_col].values

        g2 = sub[
            sub[self.group_col]
            ==
            self.group2_label
        ][self.value_col].values

        if (
            len(g1) == 0
            or
            len(g2) == 0
        ):
            return None

        q1 = self._quantile_function(
            g1,
            q_grid
        )

        q2 = self._quantile_function(
            g2,
            q_grid
        )

        return np.maximum(
            q1,
            q2
        )

    # --------------------------------------------------
    # Pooled target
    # --------------------------------------------------

    def _pooled_target(
        self,
        category,
        category_data,
        q_grid
    ):

        pool = self._pool_candidates(
            category,
            category_data.keys()
        )

        targets = []
        weights = []

        for c in pool:

            sub = category_data[c]

            n1 = (
                sub[self.group_col]
                ==
                self.group1_label
            ).sum()

            n2 = (
                sub[self.group_col]
                ==
                self.group2_label
            ).sum()

            evidence = self._evidence(
                n1,
                n2
            )

            if (
                evidence
                <
                self.min_pool_evidence
            ):
                continue

            target = (
                self._local_target(
                    sub,
                    q_grid
                )
            )

            if target is None:
                continue

            targets.append(
                target
            )

            weights.append(
                evidence
            )

        if len(weights) == 0:
            return None

        weights = np.asarray(
            weights,
            dtype=float
        )

        weights /= (
            weights.sum()
        )

        pooled = np.zeros(
            len(q_grid)
        )

        for w, target in zip(
            weights,
            targets
        ):
            pooled += w * target

        return pooled

    # --------------------------------------------------
    # Shrinkage
    # --------------------------------------------------

    def _effective_target(
        self,
        local_target,
        pooled_target,
        contrast_n
    ):

        if pooled_target is None:
            return local_target

        lam = (
            contrast_n
            /
            (
                contrast_n
                +
                self.tau
            )
        )

        return (
            lam
            *
            local_target
            +
            (
                1
                -
                lam
            )
            *
            pooled_target
        )


    def _identity_result(
        self,
        sub,
        status="insufficient_evidence"
    ):
        sub = sub.copy()
        sub["fair_value"] = sub[self.value_col]
        sub["adjustment"] = 0.0
        sub["relative_gap"] = 0.0
        sub["fairness_status"] = status

        return sub

    # --------------------------------------------------
    # Main
    # --------------------------------------------------

    def fit_transform(
        self,
        df
    ):

        df = df.copy()

        q_grid = np.linspace(
            0,
            1,
            self.quantile_grid_size
        )

        categories = {
            k: v.copy()
            for k, v in
            df.groupby(
                self.category_col
            )
        }

        outputs = []

        for (
            category,
            sub
        ) in categories.items():

            g1 = sub[
                sub[self.group_col]
                ==
                self.group1_label
            ].copy()

            g2 = sub[
                sub[self.group_col]
                ==
                self.group2_label
            ].copy()

            n1 = len(g1)
            n2 = len(g2)

            if (
                n1 == 0
                or
                n2 == 0
            ):
                outputs.append(
                    self._identity_result(
                        sub,
                        "insufficient_evidence"
                    )
                )
                continue

            contrast_n = (
                2
                *
                min(
                    n1,
                    n2
                )
            )

            local_target = (
                self._local_target(
                    sub,
                    q_grid
                )
            )

            pooled_target = (
                self._pooled_target(
                    category,
                    categories,
                    q_grid
                )
            )

            target = (
                self._effective_target(
                    local_target,
                    pooled_target,
                    contrast_n
                )
            )

            # ------------------------
            # Women
            # ------------------------

            transported = (
                self._transport_to_target(
                    g1[self.value_col].values,
                    target,
                    q_grid
                )
            )

            transported = np.maximum(
                transported,
                g1[self.value_col].values
            )

            g1["fair_value"] = (
                transported
            )

            g1["adjustment"] = (
                transported
                -
                g1[self.value_col].values
            )

            g1["fairness_status"] = (
                "pooled"
                if pooled_target is not None
                else "local"
            )

            outputs.append(g1)

            # ------------------------
            # Men
            # ------------------------

            transported = (
                self._transport_to_target(
                    g2[self.value_col].values,
                    target,
                    q_grid
                )
            )

            transported = np.maximum(
                transported,
                g2[self.value_col].values
            )

            g2["fair_value"] = (
                transported
            )

            g2["adjustment"] = (
                transported
                -
                g2[self.value_col].values
            )

            g2["fairness_status"] = (
                "pooled"
                if pooled_target is not None
                else "local"
            )

            outputs.append(g2)

        if len(outputs) == 0:

            return pd.DataFrame()

        result = pd.concat(
            outputs,
            ignore_index=True
        )

        result["relative_gap"] = (
            result["adjustment"]
            /
            result[self.value_col]
        )

        return result


if __name__ == '__main__':
    # Example 1: Group A larger
    group_a = np.array([10, 25, 40, 55, 70]) # n=5
    group_b = np.array([15, 30, 45]) # n=3

    # Usage
    idx_a, idx_b, val_a, val_b = fractional_rank_match_hungarian_simple(group_a, group_b)
    print(f"Matches: A indices {idx_a} -> B indices {idx_b}")
    print(f"Values: {list(zip(val_a, val_b))}")

    print("the compact matches are:")
    print(get_compact_matches(group_a, group_b))

    result = fractional_rank_match_hungarian(group_a, group_b)

    print("Matched pairs (a_idx, b_idx, a_val, b_val):")
    for pair in result['matched_pairs']:
        print(f" A[{pair['index_a']}]={pair['value_a']} <-> B[{pair['index_b']}]={pair['value_b']}")

    print(f"Unmatched A indices: {result['unmatched_a']}")
    print(f"Unmatched B indices: {result['unmatched_b']}")

    # Example 2: Group B larger
    group_a = np.array([10, 25]) # n=2
    group_b = np.array([15, 30, 45, 60, 75]) # n=5

    result2 = fractional_rank_match_hungarian(group_a, group_b)
    print("\n" + "="*50 + "\n")
    print("Group B larger case:")
    for pair in result2['matched_pairs']:
        print(f" A[{pair['index_a']}]={pair['value_a']} <-> B[{pair['index_b']}]={pair['value_b']}")
    print(f"Unmatched A: {result2['unmatched_a']}")
    print(f"Unmatched B: {result2['unmatched_b']}")

    print("the compact matches are:")
    print(get_compact_matches(group_a, group_b))

    udf = pd.read_excel('../data/Underlag.xlsx')
    udf .loc[:,'Typ']      = [ '#0000ff' if v=='M' else '#ff0000'  if v=='K' else '#101010' for v in udf.loc[:,'Kön'].values.tolist() ]
    udf .loc[:,'Kön']      = [ 'Man' if v=='M' else 'Kvinna'  if v=='K' else 'Okänd' for v in udf.loc[:,'Kön'].values.tolist() ]
    udf .loc[:,'BESTA']    = [ str(v)[:2] for v in udf .loc[:,'Besta'].values.tolist() ]
    udf .loc[:,'Grad']     = [ str(v)[2] for v in udf .loc[:,'Besta'].values.tolist()  ]
    udf .loc[:,'Kategori'] = [ str(v)[3:] for v in udf .loc[:,'Besta'].values.tolist() ]
    udf .loc[:,'Index']    = udf.index.values.tolist()
    midx = np.where(udf['Kön'].values == 'Man' )[0]
    fidx = np.where(udf['Kön'].values == 'Kvinna' )[0]

    group_m = udf.iloc[midx].loc[:,'Lön'].values.tolist()
    group_f = udf.iloc[fidx].loc[:,'Lön'].values.tolist()
    matches = get_compact_matches(group_f, group_m)
    print ( matches['a_to_b_values'] )
    print ( group_f )

    for vals in zip(*matches['a_to_b_values'],*matches['a_to_b_indices']) :
        diff = vals[1]-vals[0]
        print(vals[0],vals[1],'Kvinna med ID',vals[2],'ska ha utjämningspåslag', diff if diff >0 else 0 )

    for vals in zip(*matches['b_to_a_values'],*matches['b_to_a_indices']) :
        diff = vals[1]-vals[0]
        print(vals[0],vals[1],'Man med ID',vals[2],'ska ha utjämningspåslag', diff if diff >0 else 0 )

    print("Första ordningens approximation ovan")

    adjuster = NaiveValueEquityAdjuster(
        value_col       = "Lön",
        group_col       = "Kön",
        instance1_label = "Kvinna",
        instance2_label = "Man",
        category_col    = "Besta"
    )

    report_df1 = adjuster.fit_transform(udf)
    print(report_df1)
    print("Nästa stegs approximation ovan")

    adjuster = OptimalTransportFairness(
        value_col       = "Lön",
        group_col       = "Kön",
        instance1_label = "Kvinna",
        instance2_label = "Man",
        category_col    = "Besta"
    )
    report_df2 = adjuster.fit_transform(udf)
    print(report_df2)
    print("Approximation : hantera som ett optimalt asymmetriskt transport problem, minimera höjningar")

    if True:

        fairness = (
            PooledSymmetricOptimalTransportFairness(
                value_col="Lön",
                group_col="Kön",
                category_col="Besta",
                group1_label="Kvinna",
                group2_label="Man",
                tau=10,
                min_pool_evidence=3,
                )
            )

        rdf = fairness.fit_transform(udf)
        print ( "Approximation : hantera som ett symmetriskt optimalt transport problem med poolning av små grupper, minimera höjningar")
        rdf.sort_values(by='Index').to_excel('symmetriska_lönekorrektioner_med_pooling_av_jämförbara_grupper.xlsx')

    else :

        """Enkel Symmetrisk korrektion"""
        ot1 = SymmetricOptimalTransportFairness(
            value_col       = "Lön",
            group_col       = "Kön",
            category_col    = "Besta",
            instance1_label = "Kvinna",
            instance2_label = "Man",
            regularization  = 1.0
        )
        report_df4 = ot1.fit_transform(udf)
        rdf = report_df4[0]
        print ( rdf )
        print ( "Approximation : hantera som ett symmetriskt optimalt transport problem, minimera höjningar")
        print ( rdf.describe() )
        rdf.sort_values(by='Index').to_excel('symmetriska_lönekorrektioner.xlsx')

    from counterpartner.visualise import plot_besta_corrections,plot_quality_histogram

    figs = plot_quality_histogram( rdf ,
                                  category_col  = "BESTA"                   ,
                                  x_label       = "Lönekorrektion [SEK]"    ,
                                  y_label       = "Antal individer"         )

    import matplotlib.pyplot as plt
    plt.show()
