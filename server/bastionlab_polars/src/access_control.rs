use serde::{Deserialize, Serialize};
use tonic::Status;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    safe_zone: Rule,
    unsafe_handling: UnsafeAction,
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
            safe_zone: Rule::AtLeastNOf(2, vec![self.safe_zone.clone(), other.safe_zone.clone()]),
            unsafe_handling: self.unsafe_handling.merge(other.unsafe_handling),
        }
    }

    pub fn allow_by_default() -> Self {
        Policy {
            safe_zone: Rule::True,
            unsafe_handling: UnsafeAction::Log,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Rule {
    AtLeastNOf(usize, Vec<Rule>),
    UserId(String),
    Aggregation(usize),
    True,
    False,
}

#[derive(Debug, Clone)]
pub struct Context {
    pub min_agg_size: Option<usize>,
    pub user_id: String,
}

impl Rule {
    fn verify(&self, ctx: &Context) -> Result<RuleMatch, Status> {
        match self {
            Rule::AtLeastNOf(n, sub_rules) => {
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
            Rule::UserId(expected_user_id) => Ok(if expected_user_id == &ctx.user_id {
                RuleMatch::Match
            } else {
                RuleMatch::Mismatch(String::from("UserId mismatch"))
            }),
            Rule::Aggregation(min_allowed_agg_size) => {
                Ok(match ctx.min_agg_size {
                    Some(min_agg_size) if min_agg_size >= *min_allowed_agg_size => RuleMatch::Match,
                    _ => RuleMatch::Mismatch(format!(
                        "Cannot fetch a DataFrame that does not aggregate at least {} rows of the initial dataframe uploaded by the data owner.",
                        min_allowed_agg_size
                    )),
                })
            }
            Rule::True => Ok(RuleMatch::Match),
            Rule::False => Ok(RuleMatch::Mismatch(String::from(
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
