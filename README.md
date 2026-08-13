# Counterpartner
A short code for creating matched pairs from groups.

Visit the active code via :
https://github.com/richardtjornhammar/counterpartner

Visit the published code : 
https://zenodo.org/record/3833963

Cite using :
DOI: 10.5281/zenodo.3833963

# Install with :
```
pip install counterpartner
```

# Optimal Transport Fairness methods
Below is a technical description of the OptimalTransportFairness method, focusing on its assumptions, theoretical basis, and practical interpretation as implemented in the counterpartner code. The description is grounded in the source code of the mentioned method applied to a sexist wage comparison across a dataset.

The dataset contains wages, BESTA codes (corresponding to the percieved level, character and domain of the work as established by an expert) as well as the binary sex of the worker. The dataset is not disclosed and used as an example in order to make the method concrete in terms of its function and metadata requirements.

The method is currently 1D and restricted to establishing contrast matching between binary groups.

# Methods
Below two variations are presented with the first being the baseline version for describing the issue and base methodology in a context specific fashion. The second method is the final symmetric version encourage for useage in unbiased (symmetric) assessments.

# OptimalTransportFairness: A Fairness Adjustment Framework Based on Optimal Transport
## Overview

OptimalTransportFairness is a fairness-adjustment methodology designed to identify and correct systematic value disparities between two groups while preserving the internal structure of each group. The method is inspired by one-dimensional optimal transport theory, where the objective is to transform one distribution into another using the smallest possible aggregate modification.

In the salary-equity context, the method estimates the salary increases required for members of an under-compensated group so that their salary distribution more closely matches that of a reference group, while avoiding salary reductions and limiting excessive adjustments.

## Core Assumptions

The methodology relies on several important assumptions.

### Comparability Assumption

Fairness comparisons should only be performed within comparable categories, qualifiers of the contrast under study.

Examples include:

Job classifications (BESTA codes)
Grades
Professional roles
Seniority groups

Individuals are therefore compared only against peers performing equivalent work. Cross-category comparisons are explicitly avoided.

### Distributional Fairness Assumption

The method assumes that fairness is primarily a property of the distribution rather than individual pairwise matches.

Instead of asking:

"Which man should this woman be compared to?"

the method asks:

"Where does this person lie within their group's distribution, and what would an equivalent percentile position look like in the reference group?"

This avoids arbitrary one-to-one matching and instead uses quantile correspondence.

### Monotonicity Assumption

Higher-valued individuals should remain higher-valued after adjustment. Value is directly infered through salary since it is a priori assumed to communicate the worth of the work produced.

The transport map preserves rank ordering:

lowest percentile remains lowest percentile,
median remains median,
highest percentile remains highest percentile.

Thus, the method does not reorder individuals within a group.

## Positive-Correction Principle

The framework assumes that fairness corrections should be implemented through upward adjustments only.

No individual's value is reduced. This is an assumption that can be discarded if comparisons are non-conservative. This is not the case for salaries, but could be true for other metrics outside of this concrete example.

Mathematically:

x_fair ≥ x_original

This is enforced through a positivity projection that replaces any downward transport recommendation with the original value.

# Theoretical Foundation : Optimal Transport

Optimal transport originates from the problem:

What is the least costly way to transform one distribution into another?

In one dimension, the solution is particularly elegant.

If F_A(x) and F_B(x) are the cumulative distributions of two groups, then the optimal transport map is

T(x) = F_B^{−1}(F_A(x))

which maps each percentile in Group A to the same percentile in Group B.

This mapping minimizes the Wasserstein transport cost among all monotone transformations.

The implementation computes this numerically through empirical quantiles and interpolation.

# Algorithm

For each category:

## Step 1: Extract group values

Separate observations into:

Source group (group potentially receiving adjustment)
Reference group

For example:

Women → source
Men → reference

within a specific quality (BESTA) category.

## Step 2: Compute percentile positions

Each source-group observation receives a fractional rank:

q_i = (r_i − 0.5) / n

where:

r_i = rank within the source group
n = source-group size

This converts values into percentile coordinates (fractional ranks with range (0,1) ).

## Step 3: Compute the Optimal Transport Target

The corresponding percentile is located in the reference distribution.

Example:

|Female Percentile | Female Salary | Male Percentile Match|
|------------------|---------------|----------------------|
|              10% |     38,000SEK |  Male 10th percentile|
|              50% |     47,000SEK |  Male median         |
|              90% |     63,000SEK |  Male 90th percentile|

This produces the transported value T(x_i) which represents the salary expected under distributional parity.

## Step 4: Positive Fairness Projection

If the transport map suggests a decrease T(x_i) < x_i the adjustment is discarded.

Instead:

P(x_i) = max(x_i,T(x_i))

This guarantees that nobody loses salary.

## Step 5: Conservative Regularization

Direct transport can occasionally produce large corrections.

To prevent abrupt changes, the method uses a regularized update:

x_i^{fair} = x_i + λ( P(x_i)−x_i )

where

0<λ≤1

is the regularization parameter.

In the implementation:

λ = 0.25

by default.

This means only 25% of the theoretically required correction is applied in a single iteration.

## Step 6: Compute Adjustment Magnitude

The final adjustment becomes:

Δ_i = x_i^{fair} − x_i

and the relative gap:

g_i = Δ_i/x_i 

which can be used for reporting and prioritization.

Why Optimal Transport?

Compared with pairwise matching approaches, optimal transport offers several advantages:

Global Consistency

All observations contribute simultaneously to the fairness estimate.

Rank Preservation

Internal merit ordering remains unchanged.

Minimal Aggregate Change

The transport map corresponds to the minimum-cost distributional transformation in one dimension.

# Robustness

No arbitrary matching choices are required.

# Scalability

Computational complexity is dominated by sorting and interpolation, making it practical for large datasets.

# Interpretation

Conceptually, the method treats fairness correction as a distribution alignment problem:

"What is the smallest upward redistribution required so that members of the source group occupy the same percentile positions as members of the reference group within comparable categories?"

Rather than enforcing equality of means, medians, or individual matches, the method seeks quantile parity while respecting three constraints:

Compare only comparable individuals.
Never reduce anyone's value.
Apply adjustments conservatively.

The result is a mathematically principled fairness framework rooted in the theory of Wasserstein optimal transport, producing interpretable and economically conservative adjustment recommendations.

# Overall Assessment

## The strongest aspects are:

1. No arbitrary pair matching.
2. Rank preservation.
3. Transparent mathematics.
4. Minimal-change property (within the chosen fairness definition).
5. Easy to explain and audit.

## The most serious weaknesses are:

1. Distribution differences are treated as evidence of unfairness.
2. Validity depends entirely on the quality of the comparability metadata (BESTA).
3. No uncertainty estimation or significance testing.
4. The reference distribution is assumed fair.
5. Quantile parity is a normative choice rather than an objectively correct fairness criterion.

# The major flaw

For the salary example we do not want to enfoce sexism and the approach places one group as the reference. This is inherently sexist, but we can alleviate the issue by extending our approach to act symmetrically.


# Pooled Symmetric Optimal Transport Fairness

This methods solves some of the technical issues relating to non-symmetric treatment of the problem and constitutes a self-consistent equalizing methodolgy between the group instances. It is agnostic to which group is dominant. However much of the methodology is directly borrowed from the above method and as such common details are omitted from the description.

## Overview: "Pooled Symmetric Optimal Transport Fairness" is a distributional fairness framework designed to identify and correct systematic disparities between two groups while preserving rank ordering, avoiding metric reductions, and remaining robust in categories with limited sample sizes.

The method extends classical one-dimensional optimal transport (OT) by introducing:

1. Symmetric fairness targets rather than selecting one group as the reference.
2. Positive-only corrections, ensuring that no individual loses value.
3. Hierarchical pooling across comparable categories, allowing statistically weak groups to borrow information from similar groups.
4. Shrinkage regularization, which balances local evidence against pooled evidence.

The framework is applicable to absolute metric analyses (such as salaries), but may also be used for any scalar metric where fairness is interpreted as a distributional alignment problem.

## Category Structure

The method assumes that each observation belongs to a category represented by a structured code: XXC… where XX represents the work type, C represents complexity, the remaining characters represent expert-defined refinements. Categories are assumed to represent self similar groupings.

The codings are parsed as:

|Component | Meaning        |
|----------|----------------|
|First two characters | Type|
|Third character | Complexity|
|Remaining characters | Expert-defined tail|

The complete category code defines the local comparison group, while only the pair (work type,complexity) is used to identify comparable categories for pooling. Pooling is done in order to collect small groups into larger comparable ones when local evaluation fails due to lack of evidence size in the comparison.

The method is a symmetric, rank-preserving, one-dimensional Wasserstein fairness adjustment in which the common target distribution is the minimal upper quantile envelope of the two group distributions, subject to a no-downward-adjustment constraint.

## Local Fairness Target

The method first constructs a local fairness target for each category.

For every percentile:

Q_local(p) = max(Q_F(p),Q_M(p)).

This is the upper quantile envelope.

The target distribution therefore lies above both observed distributions at every percentile. The interpretation then means that at each percentile position for a metric (such as salary) for a group contrast axis such as men and women:

if women earn more, the female quantile defines the target,
if men earn more, the male quantile defines the target.

Thus Q_local is the smallest distribution that dominates both groups pointwise.

## Evidence Measure

The reliability of a category depends on the availability of observations from both groups.

The evidence score is:

E_c = min(n_F,n_M),

where n_F and n_M are the group sample sizes.

A category with balanced representation receives more weight than a category dominated by one group.

## Pooling Across Comparable Categories

Small categories often provide unstable quantile estimates. To improve robustness, the method pools information from categories sharing the same (type,complexity).

For category c, define the pool:

P(c) = { r : r != c, (type_r, complexity_r) = ( type_c, complexity_c) }.

Only categories with sufficient evidence are included.

## Pooled Fairness Target

For each pooled category r, a local fairness target Q_r(p) is computed.

These are combined using evidence weights:

w_r = E_r / ∑_j E_j 

The pooled target becomes:

Q_pool(p) =  ∑_{r ∈ P(c)}  w_r Q_r(p).

This pooled target represents the fairness structure observed across comparable categories.

### Shrinkage Estimation

The final target is a weighted combination of:

1. the local fairness target,
2. the pooled fairness target.

Define the contrast sample size N_c = 2 min(n_F,n_M).

The shrinkage coefficient is λ_c = N_c / ( N_c + τ ) where τ>0 which is a user-defined regularization parameter.

The effective target becomes: Q^{∗}(p) = λ_c Q_local(p) + (1−λ_c) Q_pool(p)

#### Interpretation

Large categories N_c ≫ τ yield λ_c ≈ 1, so the estimate is driven primarily by local data.

Small categories N_c ≪ τ yield λ_c ≈ 0 and therefore borrow strength from comparable categories.

This is analogous to empirical-Bayes shrinkage.

### Optimal Transport Mapping
Then makes use of the fractional rankings p_i of salary x_i and forms the transport relation T(x_i) = Q^{∗}(p_i)

### Positive Projection

The transport recommendation is projected onto the admissible set T^{+}(x_i) = max(x_i,T(x_i)). Thus no individual receives a negative adjustment.

In practice, I would view the method as a distributional disparity correction framework rather than a complete fairness detector.

### Final Adjustment

The fairness correction is Δ_i = T^{+}(x_i)−x_i. Where the relative correction is R_i = Δ_i / x_i. Then the adjusted value becomes:

x_i^{fair} = x_i + Δ_i

## Theoretical Properties

The method satisfies the following properties.

1. Symmetry - Neither group is treated as the reference group. Both groups are evaluated relative to the same target distribution.
2. Rank Preservation - Ordering within each group is preserved.
3. Positive Corrections Only - No individual receives a reduction.
4. Local Comparability - Fairness estimation is performed within categories.
5. Pooling occurs only among predefined comparable categories (set with τ)
6. Statistical Stabilization - Small groups borrow information from similar categories through evidence-weighted pooling.
7. Distributional Fairness - The method equalizes quantile positions rather than means, medians, or arbitrary pairwise matches.

Conceptually, the method asks: What is the smallest upward adjustment required so that both groups could occupy a common fairness distribution while preserving rank structure and respecting category-specific comparability constraints?

