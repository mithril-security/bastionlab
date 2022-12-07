use linfa::{traits::Fit, DatasetBase};
use linfa_trees::{DecisionTree, DecisionTreeParams, Result, SplitQuality};
use ndarray::{ArrayBase, Ix1, Ix2, OwnedRepr};

pub fn decision_trees(
    train: DatasetBase<ArrayBase<OwnedRepr<f64>, Ix2>, ArrayBase<OwnedRepr<usize>, Ix1>>,
    split_quality: SplitQuality,
    max_depth: Option<usize>,
    min_weight_split: f32,
    min_weight_leaf: f32,
    min_impurity_decrease: f64,
) -> Result<DecisionTree<f64, usize>> {
    let reg = DecisionTreeParams::new()
        .split_quality(split_quality)
        .max_depth(max_depth)
        .min_weight_split(min_weight_split)
        .min_weight_leaf(min_weight_leaf)
        .min_impurity_decrease(min_impurity_decrease);

    let model = reg.fit(&train)?;

    Ok(model)
}
