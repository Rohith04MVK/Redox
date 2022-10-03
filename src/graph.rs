//! Defining things related to `ag::Graph`.

use crate::tensor::{Tensor, TensorInternal};

use crate::tensor_ops as T;
use crate::variable::{VariableID, VariableNamespace};
use crate::{Float, FxHashMap, NdArray, VariableEnvironment};

use std::cell::RefCell;
use std::cell::{Ref, RefMut};
use std::fmt;
use std::ops::Deref;

pub type TensorID = usize;

pub const NUM_NODES_WARN: usize = 50_000;
pub const NUM_NODES_CRITICAL: usize = 500_000;

impl<'t, 'g, F: Float> Graph<F> {
    #[inline]
    pub(crate) fn install(&'g self, mut node: TensorInternal<F>) -> TensorID {
        let mut inner = self.node_set.borrow_mut();
        let id = inner.len();
        if id == NUM_NODES_WARN {
            eprintln!(
                "Too many tensors in this graph: {}. \
            Use Graph::clear, or stop using loops in the VariableEnvironment::run block",
                NUM_NODES_WARN
            )
        }
        if id > NUM_NODES_CRITICAL {
            panic!(
                "Maximum graph size exceeded: {}. \
            Use Graph::clear, or stop using loops in the VariableEnvironment::run block",
                NUM_NODES_CRITICAL
            )
        }
        node.id = id;
        inner.push(node);
        id
    }

    #[inline(always)]
    pub(crate) fn access_inner(&self, i: TensorID) -> Ref<TensorInternal<F>> {
        let borrow = self.node_set.borrow();
        Ref::map(borrow, |t| &t[i])
    }

    #[inline(always)]
    pub(crate) fn access_inner_mut(&self, i: TensorID) -> RefMut<TensorInternal<F>> {
        let borrow = self.node_set.borrow_mut();
        RefMut::map(borrow, |t| &mut t[i])
    }

    #[inline(always)]
    pub(crate) fn tensor(&'g self, id: TensorID) -> Tensor<'g, F> {
        Tensor { id, graph: self }
    }
}

impl<T: Float> fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let set = &*self.node_set.borrow();
        let mut buf = format!("graph size: {}\n", set.len());
        for node in set {
            buf += format!("{}\n", node).as_str();
        }
        write!(f, "{}", buf)
    }
}

/// Creates and runs a computation graph.
///
/// See [Context].
pub fn run<F, FN, R>(f: FN) -> R
where
    F: Float,
    FN: FnOnce(&mut Context<F>) -> R,
{
    let env_handle = &mut VariableEnvironment::new();
    let graph_internal = Graph {
        node_set: RefCell::new(Vec::with_capacity(512)),
        variable2node: RefCell::new(FxHashMap::default()),
    };
    let mut g = Context {
        env_handle,
        inner: graph_internal,
    };
    f(&mut g)
}
