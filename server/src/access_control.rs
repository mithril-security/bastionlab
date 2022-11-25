use serde::{Deserialize, Serialize};
use tonic::Status;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    fetch: PolicyEntry,
}

impl Policy {
    pub fn verify_fetch(&self, ctx: &Context) -> Result<PolicyAction, Status> {
        self.fetch.run_checks(ctx)
    }

    pub fn merge(&self, other: &Self) -> Self {
        Policy {
            fetch: self.fetch.merge(&other.fetch),
        }
    }

    pub fn allow_by_default() -> Self {
        Policy {
            fetch: PolicyEntry::allow_by_default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyEntry {
    accept: Rule,
    approval: Rule,
}

impl PolicyEntry {
    fn run_checks(&self, ctx: &Context) -> Result<PolicyAction, Status> {
        Ok(match self.accept.verify(ctx)? {
            RuleMatch::Match => PolicyAction::Accept,
            RuleMatch::Mismatch(reason) => match self.approval.verify(ctx)? {
                RuleMatch::Match => PolicyAction::Approval(reason),
                RuleMatch::Mismatch(reason) => PolicyAction::Reject(reason),
            },
        })
    }

    pub fn merge(&self, other: &Self) -> Self {
        PolicyEntry {
            accept: Rule::AtLeastNOf(2, vec![self.accept.clone(), other.accept.clone()]),
            approval: Rule::AtLeastNOf(2, vec![self.approval.clone(), other.approval.clone()]),
        }
    }

    pub fn allow_by_default() -> Self {
        PolicyEntry {
            accept: Rule::True,
            approval: Rule::True,
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
                    Some(min_agg_size) if min_agg_size <= *min_allowed_agg_size => RuleMatch::Match,
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

#[derive(Debug, Clone)]
pub enum PolicyAction {
    Accept,
    Reject(String),
    Approval(String),
}
