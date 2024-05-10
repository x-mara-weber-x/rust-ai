use crate::matrices::simd::SimdMatrix;
use std::fmt::Debug;

// Common 3D graphics types
pub type Scalar<T> = SimdMatrix<T, 1, 1>;
pub type Vector2<T> = SimdMatrix<T, 2, 1>;
pub type Vector2Transposed<T> = SimdMatrix<T, 1, 2>;
pub type Vector3<T> = SimdMatrix<T, 3, 1>;
pub type Vector3Transposed<T> = SimdMatrix<T, 1, 3>;
pub type Vector4<T> = SimdMatrix<T, 4, 1>;
pub type Vector4Transposed<T> = SimdMatrix<T, 1, 4>;
pub type Matrix2x2<T> = SimdMatrix<T, 2, 2>;
pub type Matrix2x3<T> = SimdMatrix<T, 2, 3>;
pub type Matrix3x2<T> = SimdMatrix<T, 3, 2>;
pub type Matrix3x3<T> = SimdMatrix<T, 3, 3>;
pub type Matrix4x4<T> = SimdMatrix<T, 4, 4>;
pub type Matrix3x4<T> = SimdMatrix<T, 3, 4>;
pub type Matrix4x3<T> = SimdMatrix<T, 3, 4>;

pub type SimdVector<T, const DIMENSION: usize> = SimdMatrix<T, DIMENSION, 1>;
pub type SimdVectorTransposed<T, const DIMENSION: usize> = SimdMatrix<T, 1, DIMENSION>;
