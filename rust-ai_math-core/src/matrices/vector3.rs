use crate::matrices::aliases::{Matrix3x3, Vector2, Vector3};
use crate::matrices::simd::SimdCell;
use crate::matrices::vector::Vector;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

// Methods that only make sense for a 3D vector.
// pub trait Vector3Utilities<T: CellBase>: Vector<T> {
//     fn up() -> Self {
//         Self::new(&[T::zero(), T::one(), T::zero()])
//     }
//     fn down() -> Self {
//         Self::new(&[T::zero(), -T::one(), T::zero()])
//     }
//     fn back() -> Self {
//         Self::new(&[T::zero(), T::zero(), -T::one()])
//     }
//     fn forward() -> Self {
//         Self::new(&[T::zero(), T::zero(), T::one()])
//     }
//     fn left() -> Self {
//         Self::new(&[-T::one(), T::zero(), T::zero()])
//     }
//     fn right() -> Self {
//         Self::new(&[T::one(), T::zero(), T::zero()])
//     }
//
//     fn x(&self) -> T {
//         self[0]
//     }
//
//     fn y(&self) -> T {
//         self[1]
//     }
//
//     fn z(&self) -> T {
//         self[2]
//     }
//
//     fn cross_product(&self, other: &Self) -> Self {
//         Self::new(&[
//             self.y() * other.z() - self.z() * other.y(),
//             -(self.x() * other.z() - self.z() * other.x()),
//             self.x() * other.y() - self.y() * other.x(),
//         ])
//     }
// }

//impl<T: CellBase> Vector3Utilities<T> for Vector3<T> {}
