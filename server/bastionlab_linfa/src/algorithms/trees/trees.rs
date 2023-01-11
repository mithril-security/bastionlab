use linfa::{traits::Fit, DatasetBase};
use linfa_trees::{DecisionTreeParams, SplitQuality};

pub fn decision_trees(
    split_quality: SplitQuality,
    max_depth: Option<usize>,
    min_weight_split: f32,
    min_weight_leaf: f32,
    min_impurity_decrease: f64,
) -> DecisionTreeParams<f64, usize> {
    let reg = DecisionTreeParams::new()
        .split_quality(split_quality)
        .max_depth(max_depth)
        .min_weight_split(min_weight_split)
        .min_weight_leaf(min_weight_leaf)
        .min_impurity_decrease(min_impurity_decrease);

    reg
}
