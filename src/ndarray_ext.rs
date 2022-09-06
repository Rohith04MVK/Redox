extern crate ndarray;

pub type NdArray = ndarray::Array<f32, ndarray::IxDyn>;

pub type NdArrayView<'a> = ndarray::ArrayView<'a, f32, ndarray::IxDyn>;

#[inline]
pub fn arr_to_shape(arr: &NdArray) -> Vec<usize> {
    arr.iter().map(|&a| a as usize).collect::<Vec<_>>()
}

#[doc(hidden)]
#[inline]
pub fn expand_dims_view<'a>(x: NdArrayView<'a>, axis: usize) -> NdArrayView<'a> {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn expand_dims(x: NdArray, axis: usize) -> NdArray {
    let mut shape = x.shape().to_vec();
    shape.insert(axis, 1);
    x.into_shape(shape).unwrap()
}

#[doc(hidden)]
#[inline]
pub fn roll_axis(arg: &mut NdArray, to: ndarray::Axis, from: ndarray::Axis) {
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
