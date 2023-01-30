use std::{ops::Deref, rc::Rc, cell::RefCell};

use polars::prelude::*;

use crate::{
    errors::BastionLabPolarsError,
    visit::{self, Visitor},
};

#[derive(Debug, Clone)]
pub struct StateNode<S> {
    pub children: StateNodeChildren<Self>,
    pub state: RefCell<Option<S>>,
}

#[derive(Debug, Clone)]
pub enum StateNodeChildren<T> {
    Empty,
    Unary(Rc<T>),
    Binary(Rc<T>, Rc<T>),
}

impl<S> StateNode<S> {
    pub fn new(state: S) -> Self {
        StateNode {
            children: StateNodeChildren::Empty,
            state: RefCell::new(Some(state)),
        }
    }

    pub fn empty() -> Self {
        StateNode {
            children: StateNodeChildren::Empty,
            state: RefCell::new(None),
        }
    }

    pub fn unary(node: Rc<Self>) -> Self {
        StateNode {
            children: StateNodeChildren::Unary(node),
            state: RefCell::new(None),
        }
    }

    pub fn binary(left: Rc<Self>, right: Rc<Self>) -> Self {
        StateNode {
            children: StateNodeChildren::Binary(left, right),
            state: RefCell::new(None),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StateTree<S> {
    root: Rc<StateNode<S>>,
    ptr: Rc<StateNode<S>>,
    prev: Vec<Rc<StateNode<S>>>,
    backtracking: bool,
}

impl<S> Deref for StateTree<S> {
    type Target = StateNode<S>;
    fn deref(&self) -> &Self::Target {
        self.ptr.deref()
    }
}

impl<S> StateTree<S> {
    pub fn new(node: StateNode<S>) -> Self {
        let root = Rc::new(node);
        StateTree {
            prev: Vec::new(),
            ptr: Rc::clone(&root),
            backtracking: false,
            root,
        }
    }

    pub fn next(&mut self) {
        self.backtracking = false;
        match &self.ptr.children {
            StateNodeChildren::Empty => (),
            StateNodeChildren::Unary(child) => {
                self.prev.push(Rc::clone(&self.ptr));
                self.ptr = Rc::clone(child);
            }
            StateNodeChildren::Binary(left_child, right_child) => {
                let child = if self.backtracking {
                    right_child
                } else {
                    left_child
                };
                self.prev.push(Rc::clone(&self.ptr));
                self.ptr = Rc::clone(child);
            }
        }
    }

    pub fn prev(&mut self) {
        self.backtracking = true;
        if let Some(node) = self.prev.pop() {
            self.ptr = node;
        }
    }

    pub fn try_unwrap(self) -> Result<StateNode<S>, BastionLabPolarsError> {
        drop(self.ptr);
        drop(self.prev);
        
        Rc::try_unwrap(self.root).map_err(|_| BastionLabPolarsError::BadState)
    }
}

#[derive(Debug, Clone)]
pub struct StateTreeBuilder<S> {
    main_stack: Vec<Rc<StateNode<S>>>,
    initial_states: bool,
}

impl<S> StateTreeBuilder<S> {
    pub fn new() -> Self {
        StateTreeBuilder {
            main_stack: Vec::new(),
            initial_states: false,
        }
    }

    pub fn from_initial_states(initial_states: Vec<S>) -> Self {
        StateTreeBuilder {
            main_stack: initial_states
                .into_iter()
                .map(|state| Rc::new(StateNode::new(state)))
                .collect(),
            initial_states: true,
        }
    }

    pub fn state_tree(mut self) -> Result<StateTree<S>, BastionLabPolarsError> {
        Rc::try_unwrap(
            self.main_stack
                .pop()
                .ok_or(BastionLabPolarsError::BadState)?,
        )
        .map(|node| StateTree::new(node))
        .map_err(|_| BastionLabPolarsError::BadState)
    }
}

impl<S> Visitor for StateTreeBuilder<S> {
    fn visit_logical_plan(
        &mut self,
        node: &LogicalPlan,
    ) -> Result<(), crate::errors::BastionLabPolarsError> {
        visit::visit_logical_plan(self, node)?;

        match node {
            LogicalPlan::DataFrameScan { .. } if !self.initial_states => {
                self.main_stack.push(Rc::new(StateNode::empty()))
            }

            LogicalPlan::Selection { .. }
            | LogicalPlan::Cache { .. }
            | LogicalPlan::LocalProjection { .. }
            | LogicalPlan::Projection { .. }
            | LogicalPlan::Aggregate { .. }
            | LogicalPlan::HStack { .. }
            | LogicalPlan::Distinct { .. }
            | LogicalPlan::Sort { .. }
            | LogicalPlan::Explode { .. }
            | LogicalPlan::Slice { .. }
            | LogicalPlan::Melt { .. }
            | LogicalPlan::MapFunction { .. } => {
                let node = self
                    .main_stack
                    .pop()
                    .ok_or(BastionLabPolarsError::EmptyStack)?;
                self.main_stack.push(Rc::new(StateNode::unary(node)));
            }

            LogicalPlan::Join { .. } => {
                let right_node = self
                    .main_stack
                    .pop()
                    .ok_or(BastionLabPolarsError::EmptyStack)?;
                let left_node = self
                    .main_stack
                    .pop()
                    .ok_or(BastionLabPolarsError::EmptyStack)?;

                self.main_stack
                    .push(Rc::new(StateNode::binary(left_node, right_node)));
            }
            _ => (),
        }

        Ok(())
    }
}
