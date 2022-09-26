#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
/// re-exported for convenience and version-compatibility
pub extern crate ndarray;

#[cfg(all(feature = "blas", feature = "intel-mkl"))]
extern crate intel_mkl_src;

#[cfg(all(feature = "blas", not(feature = "intel-mkl")))]
extern crate blas_src;
#[cfg(feature = "blas")]
extern crate cblas_sys;

extern crate libc;
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
/// re-exported for convenience and version-compatibility
pub extern crate rand;
extern crate rand_distr;
extern crate rand_xorshift;
extern crate rayon;
extern crate rustc_hash;
extern crate serde_json;
pub(crate) extern crate smallvec;
extern crate uuid;
#[macro_use]
extern crate serde_derive;
extern crate approx;
extern crate special;

use std::fmt;
use serde::Deserialize;
use serde::Serialize;

/// A primitive type in this crate, which is actually a decorated `num_traits::Float`.
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + Serialize
    + Deserialize<'static>
    + 'static
{
}

#[doc(hidden)]
/// Internal trait.
pub trait Int:
    num::Integer
    + num_traits::NumAssignOps
    + num_traits::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + Sized
    + Serialize
    + Deserialize<'static>
    + 'static
{
}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + Serialize
        + Deserialize<'static>
        + 'static
{
}

impl<T> Int for T where
    T: num::Integer
        + num_traits::NumAssignOps
        + num_traits::ToPrimitive
        + Copy
        + Send
        + Sync
        + fmt::Display
        + Sized
        + Serialize
        + Deserialize<'static>
        + 'static
{
}