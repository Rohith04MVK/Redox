use ndarray;

// expose array_gen
use crate::Float;

/// alias for `ndarray::Array<T, IxDyn>`
pub type NdArray<T> = ndarray::Array<T, ndarray::IxDyn>;

/// alias for `ndarray::ArrayView<T, IxDyn>`
pub type NdArrayView<'a, T> = ndarray::ArrayView<'a, T, ndarray::IxDyn>;

/// alias for `ndarray::ArrayViewMut<T, IxDyn>`
pub type NdArrayViewMut<'a, T> = ndarray::ArrayViewMut<'a, T, ndarray::IxDyn>;

use rand_xorshift;

#[inline]
/// This works well only for small arrays
pub(crate) fn as_shape<T: Float>(x: &NdArrayView<T>) -> Vec<usize> {
    x.iter().map(|a| a.to_usize().unwrap()).collect()
}

#[inline]
pub(crate) fn expand_dims<T: Float>(x: NdArray<T>, axis: usize) -> NdArray<T> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[inline]
pub(crate) fn roll_axis<T: Float>(arg: &mut NdArray<T>, to: ndarray::Axis, from: ndarray::Axis) {
    let i = to.index();
    let mut j = from.index();
    if j > i {
        while i != j {
            arg.swap_axes(i, j);
            j -= 1;
        }
    } else {
        while i != j {
            arg.swap_axes(i, j);
            j += 1;
        }
    }
}

#[inline]
pub(crate) fn normalize_negative_axis(axis: isize, ndim: usize) -> usize {
    if axis < 0 {
        (ndim as isize + axis) as usize
    } else {
        axis as usize
    }
}

#[inline]
pub(crate) fn normalize_negative_axes<T: Float>(axes: &NdArrayView<T>, ndim: usize) -> Vec<usize> {
    let mut axes_ret: Vec<usize> = Vec::with_capacity(axes.len());
    for &axis in axes.iter() {
        let axis = if axis < T::zero() {
            (T::from(ndim).unwrap() + axis).to_usize().unwrap()
        } else {
            axis.to_usize().unwrap()
        };
        axes_ret.push(axis);
    }
    axes_ret
}

#[inline]
pub(crate) fn sparse_to_dense<T: Float>(arr: &NdArrayView<T>) -> Vec<usize> {
    let mut axes: Vec<usize> = vec![];
    for (i, &a) in arr.iter().enumerate() {
        if a == T::one() {
            axes.push(i);
        }
    }
    axes
}

#[allow(unused)]
#[inline]
pub(crate) fn is_fully_transposed(strides: &[ndarray::Ixs]) -> bool {
    let mut ret = true;
    for w in strides.windows(2) {
        if w[0] > w[1] {
            ret = false;
            break;
        }
    }
    ret
}

#[inline]
pub(crate) fn copy_if_not_standard<T: Float>(x: &NdArrayView<T>) -> Option<NdArray<T>> {
    if !x.is_standard_layout() {
        Some(deep_copy(x))
    } else {
        None
    }
}

#[inline]
pub(crate) fn deep_copy<T: Float>(x: &NdArrayView<T>) -> NdArray<T> {
    let vec = x.iter().cloned().collect::<Vec<_>>();
    NdArray::from_shape_vec(x.shape(), vec).unwrap()
}

#[inline]
pub(crate) fn scalar_shape<T: Float>() -> NdArray<T> {
    // safe unwrap
    NdArray::from_shape_vec(ndarray::IxDyn(&[0]), vec![]).unwrap()
}

#[inline]
pub(crate) fn is_scalar_shape(shape: &[usize]) -> bool {
    shape == [] || shape == [0]
}
