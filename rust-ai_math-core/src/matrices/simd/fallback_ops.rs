use crate::matrices::simd::simd_matrix::{SimdCell, SimdMatrix};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::simd::{LaneCount, SupportedLaneCount};

/// This allows us to avoid collisions by providing generic stubs even though we have specialized
/// implementations. We need the stubs to allow for generic algorithms. We do not want to implement
/// every single algorithm for each numeric type lol. This only matters for the fundamental operations
/// that directly interact with SIMD intrinsics. Everything a layer above will be able to solely
/// work with generics and `CellBase` with the caveat that if you invoke a SIMD implementation with
/// an unsupported type, the code will panic. However, we can gate that at the top-level and fallback
/// to a different backing type altogether in these cases. This is mostly to just make the compiler happy.
pub(crate) auto trait ProvideGenericFallback {}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    MulAssign<T> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: T) {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Mul<T>
    for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    AddAssign<T> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: T) {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Add<T>
    for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    DivAssign<T> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn div_assign(&mut self, rhs: T) {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    SubAssign<T> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: T) {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Sub<T>
    for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Div<T>
    for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    SubAssign<SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn sub_assign(&mut self, rhs: Self) {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Sub
    for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    Add<SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        todo!()
    }
}

impl<T: SimdCell + ProvideGenericFallback, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    AddAssign<SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn add_assign(&mut self, rhs: Self) {
        todo!()
    }
}
