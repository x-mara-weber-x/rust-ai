use crate::matrices::simd::fallback_ops::ProvideGenericFallback;
use crate::matrices::simd::simd_matrix::{SimdCell, SimdMatrix};
use crate::matrices::vector::Vector;
use crate::meta::predicate::{Predicate, Satisfied};
use float_cmp::ApproxEq;
use num_traits::real::Real;
use num_traits::{NumAssign, NumAssignRef, One, Signed, Zero};
use quickcheck::{empty_shrinker, Arbitrary, Gen};
use std::any::type_name;
use std::cell::Cell;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{LaneCount, Simd, SimdCast, SimdElement, SupportedLaneCount};

impl !ProvideGenericFallback for f32 {}
impl !ProvideGenericFallback for f64 {}
impl !ProvideGenericFallback for i8 {}
impl !ProvideGenericFallback for u8 {}
impl !ProvideGenericFallback for i16 {}
impl !ProvideGenericFallback for u16 {}
impl !ProvideGenericFallback for i32 {}
impl !ProvideGenericFallback for u32 {}
impl !ProvideGenericFallback for i64 {}
impl !ProvideGenericFallback for u64 {}

macro_rules! math_ops_element_wise {
    ( ($($numeric_type:ty)*), $op_trait:ident, $op:ident) => {
        $(
            impl<const ROW_COUNT: usize, const COLUMN_COUNT: usize> $op_trait<SimdMatrix<$numeric_type, ROW_COUNT, COLUMN_COUNT>>
                for SimdMatrix<$numeric_type, ROW_COUNT, COLUMN_COUNT>
            where
                LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
            {
                type Output = Self;

                fn $op(self, other: Self) -> Self::Output {
                    Self {
                        simd: self.simd.$op(other.simd)
                    }
                }
            })*
    }
}
math_ops_element_wise! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Add, add}
math_ops_element_wise! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Sub, sub}

macro_rules! math_ops_by_scalar {
    ( ($($numeric_type:ty)*), $op_trait:ident, $op:ident) => {
        $(
            impl<const ROW_COUNT: usize, const COLUMN_COUNT: usize> $op_trait<$numeric_type>
                for SimdMatrix<$numeric_type, ROW_COUNT, COLUMN_COUNT>
            where
                LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
            {
                type Output = Self;

                fn $op(self, other: $numeric_type) -> Self::Output {
                    let scalar: Self::SIMD = Simd::splat(other);

                    Self {
                        simd: self.simd.$op(scalar)
                    }
                }
            })*
    }
}

math_ops_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Add, add}
math_ops_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Sub, sub}
math_ops_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Mul, mul}
math_ops_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Div, div}
math_ops_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), Rem, rem}

macro_rules! math_ops_assign_element_wise {
    ( ($($numeric_type:ty)*), $op_trait:ident, $target:ident $op:ident $other:ident) => {
        $(
            impl<const ROW_COUNT: usize, const COLUMN_COUNT: usize> $op_trait<SimdMatrix<$numeric_type, ROW_COUNT, COLUMN_COUNT>>
                for SimdMatrix<$numeric_type, ROW_COUNT, COLUMN_COUNT>
            where
                LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
            {
                fn $op(&mut self, $other: Self) {
                    self.simd.$op($other.simd);
                }
            })*
    }
}
math_ops_assign_element_wise! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), AddAssign, target add_assign other}
math_ops_assign_element_wise! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), SubAssign, target sub_assign other}

macro_rules! math_ops_assign_by_scalar {
    ( ($($numeric_type:ty)*), $op_trait:ident, $op:ident) => {
        $(
            impl<const ROW_COUNT: usize, const COLUMN_COUNT: usize> $op_trait<$numeric_type>
                for SimdMatrix<$numeric_type, ROW_COUNT, COLUMN_COUNT>
            where
                LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
            {
                fn $op(&mut self, other: $numeric_type) {
                    let scalar: Self::SIMD = Simd::splat(other);
                    self.simd.$op(scalar);
                }
            })*
    }
}
math_ops_assign_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), AddAssign, add_assign}
math_ops_assign_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), SubAssign, sub_assign}
math_ops_assign_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), MulAssign, mul_assign}
math_ops_assign_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), DivAssign, div_assign}
math_ops_assign_by_scalar! {(f32 f64 i8 u8 i16 u16 i32 u32 i64 u64), RemAssign, rem_assign}
