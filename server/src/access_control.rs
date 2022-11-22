use crate::composite_plan::CompositePlan;

use serde::{Deserialize, Serialize};
use tonic::Status;

// use std::ops::{BitOr, BitAnd, BitOrAssign, BitAndAssign};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Policy {
    fetch: PolicyEntry,
}

impl Policy {
    pub fn verify_fetch(
        &self,
        plan: &CompositePlan,
        user_id: &str,
    ) -> Result<PolicyAction, Status> {
        self.fetch.run_checks(plan, user_id)
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
    fn run_checks(&self, plan: &CompositePlan, user_id: &str) -> Result<PolicyAction, Status> {
        Ok(match self.accept.verify(plan, user_id)? {
            RuleMatch::Match => PolicyAction::Accept,
            RuleMatch::Mismatch(reason) => match self.approval.verify(plan, user_id)? {
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

impl Rule {
    fn verify(&self, plan: &CompositePlan, user_id: &str) -> Result<RuleMatch, Status> {
        match self {
            Rule::AtLeastNOf(n, sub_rules) => {
                let mut m = 0;
                let mut failed = String::new();
                for (i, rule) in sub_rules.iter().enumerate() {
                    match rule.verify(plan, user_id)? {
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
            Rule::UserId(expected_user_id) => Ok(if expected_user_id == user_id {
                RuleMatch::Match
            } else {
                RuleMatch::Mismatch(String::from("UserId mismatch"))
            }),
            Rule::Aggregation(min_allowed_agg_size) => {
                Ok(if plan.aggregation_match(*min_allowed_agg_size)? {
                    RuleMatch::Match
                } else {
                    RuleMatch::Mismatch(format!(
                    "Cannot fetch a DataFrame that does not aggregate at least {} rows of the initial dataframe uploaded by the data owner.",
                    min_allowed_agg_size
                ))
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
