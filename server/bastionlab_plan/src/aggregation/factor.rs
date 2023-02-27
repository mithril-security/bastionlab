use std::{borrow::Borrow, fmt::Debug};

#[derive(Clone, Copy)]
pub struct AggregationFactor {
    /// K, meaning the dataframe is k-anonymous - the actual aggregation factor.
    /// One line from the input dataframe contains the info of at least `k` persons.
    /// When K is infinite, this means there is no link between the resulting value and
    /// input dataframe. (ie. literal values.)
    k: usize,
    /// One person appears in at most `duplicate_factor` lines from the input dataframe.
    duplicate_factor: usize,
}

impl Debug for AggregationFactor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.k == usize::MAX {
            write!(f, "<inf>")?;
        } else {
            write!(f, "{}", self.k)?;
        }
        write!(f, "[{}]", self.duplicate_factor)
    }
}

impl AggregationFactor {
    pub fn least_of_many(factors: impl IntoIterator<Item = impl Borrow<Self>>) -> Self {
        factors
            .into_iter()
            .fold(Self::from_constant(), |a, b| a.least(*b.borrow()))
    }

    /// Default value for litteral / constants
    pub fn from_constant() -> Self {
        Self {
            k: usize::MAX,
            duplicate_factor: 1,
        }
    }

    /// Default value for values related to input
    pub fn from_input() -> Self {
        Self {
            k: 1,
            duplicate_factor: 1,
        }
    }

    pub fn k(self) -> usize {
        self.k
    }

    pub fn duplicate_factor(self) -> usize {
        self.duplicate_factor
    }

    /// The result of a Projection which has `self` as resulting AggFactor (from analysis wrt. projected exprs)
    ///  needs to be multiplied with the last aggregation factor for this dataframe.
    pub fn after_projection(self, input_df_agg_factor: Self) -> Self {
        Self {
            k: usize::saturating_mul(self.k, input_df_agg_factor.k),
            duplicate_factor: usize::saturating_mul(
                self.duplicate_factor,
                input_df_agg_factor.duplicate_factor,
            ),
        }
    }

    /// this is the merge operation
    pub fn least(self, other: Self) -> Self {
        Self {
            k: usize::min(self.k, other.k),
            duplicate_factor: usize::max(self.duplicate_factor, other.duplicate_factor),
        }
    }

    pub fn aggregate_lines(self, nrows_to_merge: usize) -> Self {
        /// y must not be null
        fn ceil_division(x: usize, y: usize) -> usize {
            // x/y + (x % y != 0)
            usize::saturating_add(x / y, usize::from(x % y != 0))
        }
        Self {
            k: usize::saturating_mul(self.k, nrows_to_merge),
            duplicate_factor: self.duplicate_factor,
        }
    }

    pub fn duplicate_lines(self, n_duplicated: usize) -> Self {
        Self {
            k: self.k,
            duplicate_factor: usize::saturating_mul(self.duplicate_factor, n_duplicated),
        }
    }
}
