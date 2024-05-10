use crate::matrices::aliases::*;
use crate::matrices::simd::fallback_ops::ProvideGenericFallback;
use crate::matrices::vector::Vector;
use crate::meta::predicate::{Predicate, Satisfied};
use float_cmp::ApproxEq;
use num_traits::real::Real;
use num_traits::{NumAssign, NumAssignRef, One, Signed, Zero};
use quickcheck::{empty_shrinker, Arbitrary, Gen};
use std::any::type_name;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};
use std::simd::num::{SimdFloat, SimdInt};
use std::simd::{LaneCount, Simd, SimdCast, SimdElement, SupportedLaneCount};

/// This cell type provides the basics for matrix operations that don't need any concept of numbers
pub trait SimdCell:
'static
+ Debug
+ Default
+ Copy
+ Clone
+ Display
+ NumAssignRef
+ PartialEq
+ PartialOrd
+ SimdElement
{}

impl<T> SimdCell for T where
    T: 'static
    + Debug
    + Default
    + Copy
    + Clone
    + Display
    + NumAssignRef
    + PartialEq
    + PartialOrd
    + SimdElement
{}

/// This type of matrix is allocated on the stack in column major order with a compile time size.
/// As such it offers the fastest possible speed for CPU execution and is well suited to represent
/// types commonly used in 3D engines.
#[derive(Debug, Copy, Clone)]
pub struct SimdMatrix<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    pub(crate) simd: Simd<T, { (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>,
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> PartialEq for
SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn eq(&self, other: &Self) -> bool {
        // since `simd` is padded, we need to ensure to only compare the non-padded entries.
        let self_elements = self.simd.as_array().as_slice();
        let other_elements = other.simd.as_array().as_slice();
        for j in 0..COLUMN_COUNT {
            if self_elements[j >> Self::COLUMN_SHIFT..(j >> Self::COLUMN_SHIFT) + ROW_COUNT]
                != other_elements[j >> Self::COLUMN_SHIFT..(j >> Self::COLUMN_SHIFT) + ROW_COUNT] {
                return false;
            }
        }

        true
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    pub type SIMD = Simd<T, { (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>;

    pub const ROW_SIZE: usize = ROW_COUNT.next_power_of_two();
    pub const COLUMN_SIZE: usize = COLUMN_COUNT.next_power_of_two();
    pub const COLUMN_SHIFT: usize = Self::ROW_SIZE.ilog2() as usize;

    pub type Mapped<R: SimdCell> = SimdMatrix<R, ROW_COUNT, COLUMN_COUNT>;

    pub fn new(row_count: usize, column_count: usize) -> Self {
        if row_count != ROW_COUNT || column_count != COLUMN_COUNT {
            panic!(
                "Type [{:?}] can not represent an instance of a {}x{} matrix.",
                type_name::<Self>(),
                row_count,
                column_count
            );
        }

        Self::default()
    }

    pub fn row_count(&self) -> usize {
        ROW_COUNT
    }

    pub fn column_count(&self) -> usize {
        COLUMN_COUNT
    }

    pub fn element_count(&self) -> usize {
        ROW_COUNT * COLUMN_COUNT
    }

    pub fn map<F, R: SimdCell>(&self, mapper: F) -> Self::Mapped<R>
        where
            F: Fn(&T) -> R
    {
        let mut result = Self::Mapped::<R>::new(self.row_count(), self.column_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                result[(i, j)] = mapper(&self[(i, j)]);
            }
        }

        result
    }

    pub fn swap_row(&mut self, row_a: usize, row_b: usize) {
        for col in 0..self.column_count() {
            (self[(row_a, col)], self[(row_b, col)]) = (self[(row_b, col)], self[(row_a, col)]);
        }
    }

    pub fn is_close_to<M: Into<T::Margin> + Clone>(&self, other: Self, margin: M) -> bool
        where
            T: SimdCell + ApproxEq,
    {
        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                if !self[(i, j)].approx_eq(other[(i, j)], margin.clone()) {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
        LaneCount<{ (COLUMN_COUNT * ROW_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    pub type Transposed = SimdMatrix<T, COLUMN_COUNT, ROW_COUNT>;

    pub fn transposed(&self) -> Self::Transposed {
        let mut transposed = Self::Transposed::new(self.column_count(), self.row_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                transposed[(j, i)] = self[(i, j)];
            }
        }

        transposed
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Display
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
        LaneCount<{ (COLUMN_COUNT * ROW_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for j in 0..self.column_count() {
            write!(f, "[")?;
            for i in 0..self.row_count() {
                write!(f, "{},", self[(i, j)])?;
            }
            write!(f, "],")?;
        }
        write!(f, "]")
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Index<(usize, usize)>
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        debug_assert!(index.0 < ROW_COUNT);
        debug_assert!(index.1 < COLUMN_COUNT);

        &self.simd[(index.1 >> Self::COLUMN_SHIFT) + index.0]
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> IndexMut<(usize, usize)>
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        debug_assert!(index.0 < ROW_COUNT);
        debug_assert!(index.1 < COLUMN_COUNT);

        &mut self.simd[(index.1 >> Self::COLUMN_SHIFT) + index.0]
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    pub fn identity() -> SimdMatrix<T, ROW_COUNT, COLUMN_COUNT> {
        let mut result = Self::default();
        for i in 0..ROW_COUNT.min(COLUMN_COUNT) {
            result[(i, i)] = T::one();
        }
        result
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Index<usize>
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.simd[index]
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> IndexMut<usize>
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.simd[index]
    }
}

impl<T: SimdCell, const N: usize> MulAssign<SimdMatrix<T, N, N>> for SimdMatrix<T, N, N>
    where
        LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<
    T: SimdCell,
    const LEFT_ROW_COUNT: usize,
    const COMMON_COUNT: usize,
    const RIGHT_COLUMN_COUNT: usize,
> Mul<SimdMatrix<T, COMMON_COUNT, RIGHT_COLUMN_COUNT>>
for SimdMatrix<T, LEFT_ROW_COUNT, COMMON_COUNT>
    where
        LaneCount<{ (LEFT_ROW_COUNT * COMMON_COUNT).next_power_of_two() }>: SupportedLaneCount,
        LaneCount<{ (COMMON_COUNT * RIGHT_COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
        LaneCount<{ (LEFT_ROW_COUNT * RIGHT_COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Output = SimdMatrix<T, LEFT_ROW_COUNT, RIGHT_COLUMN_COUNT>;

    fn mul(self, rhs: SimdMatrix<T, COMMON_COUNT, RIGHT_COLUMN_COUNT>) -> Self::Output {
        let mut result: Self::Output = Default::default();

        for i in 0..LEFT_ROW_COUNT {
            for j in 0..RIGHT_COLUMN_COUNT {
                for x in 0..COMMON_COUNT {
                    result[(i, j)] = result[(i, j)] + self[(i, x)] * rhs[(x, j)];
                }
            }
        }

        result
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Default
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn default() -> Self {
        Self {
            simd: Simd::from_array(
                [Default::default(); (ROW_COUNT * COLUMN_COUNT).next_power_of_two()],
            ),
        }
    }
}

impl<T: SimdCell + ApproxEq, const ROW_COUNT: usize, const COLUMN_COUNT: usize> ApproxEq
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    type Margin = T::Margin;

    fn approx_eq<M: Into<Self::Margin>>(self, other: Self, margin: M) -> bool {
        let margin = margin.into();
        for i in 0..ROW_COUNT {
            for j in 0..COLUMN_COUNT {
                if !self[(i, j)].approx_eq(other[(i, j)], margin) {
                    return false;
                }
            }
        }

        true
    }
}

impl<T: SimdCell + Arbitrary, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Arbitrary
for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn arbitrary(g: &mut Gen) -> Self {
        let mut matrix = Self::default();

        for i in 0..ROW_COUNT {
            for j in 0..COLUMN_COUNT {
                matrix[(i, j)] = T::arbitrary(g);
            }
        }

        matrix
    }

    fn shrink(&self) -> Box<dyn Iterator<Item=Self>> {
        empty_shrinker()
    }
}

impl<T: SimdCell, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
From<[[T; ROW_COUNT]; COLUMN_COUNT]> for SimdMatrix<T, ROW_COUNT, COLUMN_COUNT>
    where
        LaneCount<{ (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }>: SupportedLaneCount,
{
    fn from(value: [[T; ROW_COUNT]; COLUMN_COUNT]) -> Self {
        let mut simd: Simd<T, { (ROW_COUNT * COLUMN_COUNT).next_power_of_two() }> =
            Simd::splat(T::default());
        let mut column_offset = 0;

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                simd[column_offset + j] = value[i][j];
            }

            // simd is always based on PoT and thus we need to potentially skip a few entries
            column_offset += Self::ROW_SIZE;
        }

        Self { simd }
    }
}

impl<T: SimdCell, const ROW_COUNT: usize> From<[T; ROW_COUNT]> for SimdMatrix<T, ROW_COUNT, 1>
    where
        LaneCount<{ (ROW_COUNT * 1).next_power_of_two() }>: SupportedLaneCount,
{
    fn from(value: [T; ROW_COUNT]) -> Self {
        let mut simd: Simd<T, { (ROW_COUNT * 1).next_power_of_two() }> =
            Simd::from_array([T::default(); (ROW_COUNT * 1).next_power_of_two()]);

        for i in 0..ROW_COUNT {
            simd[i] = value[i];
        }

        Self { simd }
    }
}

impl<T: SimdCell> From<T> for SimdMatrix<T, 1, 1> {
    fn from(value: T) -> Self {
        Self {
            simd: Simd::from_array([value]),
        }
    }
}
