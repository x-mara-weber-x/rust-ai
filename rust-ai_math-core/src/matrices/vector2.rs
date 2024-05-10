use crate::matrices::aliases::{Matrix3x3, Vector2, Vector3};
use crate::matrices::simd::SimdCell;
use crate::matrices::vector::Vector;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

// Methods that only make sense for a 2D vector.
// pub trait Vector2Utilities<T: CellBase>: Vector<T> {
//     fn up() -> Self {
//         Self::new(&[T::zero(), T::one()])
//     }
//     fn down() -> Self {
//         Self::new(&[T::zero(), -T::one()])
//     }
//     fn left() -> Self {
//         Self::new(&[-T::one(), T::zero()])
//     }
//     fn right() -> Self {
//         Self::new(&[T::one(), T::zero()])
//     }
//
//     fn x(&self) -> T {
//         self[0]
//     }
//
//     fn y(&self) -> T {
//         self[1]
//     }
// }

//impl<T: CellBase> Vector2Utilities<T> for Vector2<T> {}
