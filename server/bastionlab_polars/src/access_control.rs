use serde::{Deserialize, Serialize};
use tonic::Status;
use crate::{composite_plan::StatsEntry};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    safe_zone: Rule,
    unsafe_handling: UnsafeAction,
    savable: bool,
    overall_budget: PrivacyBudget,
    per_query_budget: PrivacyBudget,
    enforce_dp: bool,
}

impl Policy {
    pub fn verify(&self, ctx: &Context) -> Result<VerificationResult, Status> {
        Ok(match self.safe_zone.verify(ctx)? {
            RuleMatch::Match => VerificationResult::Safe,
            RuleMatch::Mismatch(reason) => VerificationResult::Unsafe {
                action: self.unsafe_handling,
                reason,
            },
        })
    }

    pub fn merge(&self, other: &Self) -> Self {
        Policy {
            safe_zone: Rule::AtLeastNOf {
                n: 2,
                of: vec![self.safe_zone.clone(), other.safe_zone.clone()],
            },
            unsafe_handling: self.unsafe_handling.merge(other.unsafe_handling),
            savable: self.savable && other.savable,
            overall_budget : match (self.overall_budget,other.overall_budget) {
                (PrivacyBudget::NotPrivate,PrivacyBudget::NotPrivate) => PrivacyBudget::NotPrivate,
                (PrivacyBudget::NotPrivate,PrivacyBudget::Private{budget:val})|(PrivacyBudget::Private{budget:val},PrivacyBudget::NotPrivate) => PrivacyBudget::Private{budget:val},
                (PrivacyBudget::Private{budget:val1},PrivacyBudget::Private{budget:val2}) => PrivacyBudget::Private{budget:val1.min(val2)},
            },
            per_query_budget: match (self.per_query_budget,other.per_query_budget) {
                (PrivacyBudget::NotPrivate,PrivacyBudget::NotPrivate) => PrivacyBudget::NotPrivate,
                (PrivacyBudget::NotPrivate,PrivacyBudget::Private{budget:val})|(PrivacyBudget::Private{budget:val},PrivacyBudget::NotPrivate) => PrivacyBudget::Private{budget:val},
                (PrivacyBudget::Private{budget:val1},PrivacyBudget::Private{budget:val2}) => PrivacyBudget::Private{budget:val1.min(val2)},
            },
            enforce_dp: self.enforce_dp || other.enforce_dp,
        }
    }

    pub fn allow_by_default() -> Self {
        Policy {
            safe_zone: Rule::TrueRule,
            unsafe_handling: UnsafeAction::Log,
            savable: true,
            overall_budget: PrivacyBudget::NotPrivate,
            per_query_budget: PrivacyBudget::NotPrivate,
            enforce_dp: false,
        }
    }

    pub fn check_savable(&self) -> bool {
        return self.savable;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Copy)]
#[serde(tag = "type")]
enum PrivacyBudget {
    Private{budget: f64},
    NotPrivate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Rule {
    AtLeastNOf { n: usize, of: Vec<Rule> },
    UserId { id: String },
    Aggregation { min_agg_size: usize },
    TrueRule,
    FalseRule,
}

#[derive(Debug, Clone)]
pub struct Context {
    pub stats: StatsEntry,
    pub user_id: String,
    pub df_identifier: String,
}

impl Rule {
    fn verify(&self, ctx: &Context) -> Result<RuleMatch, Status> {
        match self {
            Rule::AtLeastNOf { n, of: sub_rules } => {
                let mut m = 0;
                let mut failed = String::new();
                for (i, rule) in sub_rules.iter().enumerate() {
                    match rule.verify(ctx)? {
                        RuleMatch::Match => m += 1,
                        RuleMatch::Mismatch(reason) => {
                            failed.push_str(&format!("\nRule #{}: {}", i, reason))
                        }
                    }
                    if m >= *n {
                        return Ok(RuleMatch::Match);
                    }
                }
                Ok(RuleMatch::Mismatch(format!(
                    "Only {} subrules matched but at least {} are required.\nFailed sub rules are:{}",
                    m, n, failed
                )))
            }
            Rule::UserId {
                id: expected_user_id,
            } => Ok(if expected_user_id == &ctx.user_id {
                RuleMatch::Match
            } else {
                RuleMatch::Mismatch(String::from("UserId mismatch"))
            }),
            Rule::Aggregation {
                min_agg_size: min_allowed_agg_size,
            } => {
                let min_allowed_agg_size = *min_allowed_agg_size * ctx.stats.join_scaling;
                Ok(if ctx.stats.agg_size >= min_allowed_agg_size {
                    RuleMatch::Match
                } else {
                    RuleMatch::Mismatch(format!(
                        "Cannot fetch a result DataFrame that does not aggregate at least {} rows of DataFrame {}.",
                        min_allowed_agg_size,
                        ctx.df_identifier,
                    ))
                })
            }
            Rule::TrueRule => Ok(RuleMatch::Match),
            Rule::FalseRule => Ok(RuleMatch::Mismatch(String::from(
                "Operation denied by the data owner's policy.",
            ))),
        }
    }
}

#[derive(Debug, Clone)]
pub enum RuleMatch {
    Match,
    Mismatch(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum UnsafeAction {
    Log,
    Review,
    Reject,
}

impl UnsafeAction {
    fn merge(self, other: Self) -> Self {
        match self {
            UnsafeAction::Log => other,
            UnsafeAction::Review if other == UnsafeAction::Log => UnsafeAction::Review,
            UnsafeAction::Review => other,
            UnsafeAction::Reject => UnsafeAction::Reject,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VerificationResult {
    Safe,
    Unsafe {
        action: UnsafeAction,
        reason: String,
    },
}

impl VerificationResult {
    pub fn merge(&mut self, other: Self) {
        match (self, other) {
            (
                VerificationResult::Unsafe {
                    action: left_action,
                    reason: left_reason,
                },
                VerificationResult::Unsafe {
                    action: right_action,
                    reason: right_reason,
                },
            ) => {
                *left_action = left_action.merge(right_action);
                if *left_action == right_action {
                    *left_reason = right_reason;
                }
            }
            (x @ VerificationResult::Safe, y) => *x = y,
            _ => (),
        }
    }
}
