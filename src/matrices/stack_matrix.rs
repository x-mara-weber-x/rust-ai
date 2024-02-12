use crate::matrices::matrix::Matrix;
use crate::matrices::vector::Vector;
use crate::meta::predicate::{Predicate, Satisfied};
use std::fmt::Debug;
use std::ops::{Add, Index, IndexMut, Sub};

/// This type of matrix is allocated on the stack in column major order with a compile time size.
/// As such it offers the fastest possible speed for CPU execution and is well suited to represent
/// types commonly used in 3D engines.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct StackMatrix<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> {
    rows: [[T; ROW_COUNT]; COLUMN_COUNT],
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
{
    pub fn product<const OTHER_COLUMN_COUNT: usize, const OTHER_ROW_COUNT: usize>(
        &self,
        other: &StackMatrix<T, COLUMN_COUNT, ROW_COUNT>,
    ) -> StackMatrix<T, ROW_COUNT, OTHER_COLUMN_COUNT>
    where
        Predicate<{ COLUMN_COUNT == OTHER_ROW_COUNT }>: Satisfied,
    {
        let mut result: StackMatrix<T, ROW_COUNT, OTHER_COLUMN_COUNT> = Default::default();

        result
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Matrix
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
{
    type Cell = T;
    type Transposed = StackMatrix<T, ROW_COUNT, COLUMN_COUNT>;

    fn row_count(&self) -> usize {
        ROW_COUNT
    }

    fn column_count(&self) -> usize {
        COLUMN_COUNT
    }

    fn element_count(&self) -> usize {
        todo!()
    }

    fn transposed(&self) -> Self::Transposed {
        let mut transposed = StackMatrix::<T, ROW_COUNT, COLUMN_COUNT>::default();
        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                transposed[(i, j)] = self[(j, i)];
            }
        }

        transposed
    }

    fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();

        result
    }
}

impl<T, const ROW_COUNT: usize> Vector for StackMatrix<T, 1, ROW_COUNT>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
    Predicate<{ ROW_COUNT > 1 }>: Satisfied,
{
    type Cell = T;

    fn dimension(&self) -> usize {
        ROW_COUNT
    }
}

impl<T, const COLUMN_COUNT: usize> Vector for StackMatrix<T, COLUMN_COUNT, 1>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
    Predicate<{ COLUMN_COUNT > 1 }>: Satisfied,
{
    type Cell = T;

    fn dimension(&self) -> usize {
        COLUMN_COUNT
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Index<(usize, usize)>
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.rows[index.1][index.0]
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> IndexMut<(usize, usize)>
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.rows[index.1][index.0]
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Index<usize>
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index / COLUMN_COUNT][index % ROW_COUNT]
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> IndexMut<usize>
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index / COLUMN_COUNT][index % ROW_COUNT]
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Add
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> StackMatrix<T, COLUMN_COUNT, ROW_COUNT> {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] + other.rows[i][j]
            }
        }

        result
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Add
    for &StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Add<Output = T>,
{
    type Output = StackMatrix<T, COLUMN_COUNT, ROW_COUNT>;

    fn add(self, other: Self) -> StackMatrix<T, COLUMN_COUNT, ROW_COUNT> {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] + other.rows[i][j]
            }
        }

        result
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Sub
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self;

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] - other.rows[i][j]
            }
        }

        result
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> Default
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self {
            rows: [[Default::default(); ROW_COUNT]; COLUMN_COUNT],
        }
    }
}

impl<T, const COLUMN_COUNT: usize, const ROW_COUNT: usize> From<[[T; ROW_COUNT]; COLUMN_COUNT]>
    for StackMatrix<T, COLUMN_COUNT, ROW_COUNT>
{
    fn from(value: [[T; ROW_COUNT]; COLUMN_COUNT]) -> Self {
        Self { rows: value }
    }
}

impl<T, const ROW_COUNT: usize> From<[T; ROW_COUNT]> for StackMatrix<T, 1, ROW_COUNT> {
    fn from(value: [T; ROW_COUNT]) -> Self {
        Self { rows: [value] }
    }
}

impl<T> From<T> for StackMatrix<T, 1, 1> {
    fn from(value: T) -> Self {
        Self { rows: [[value]] }
    }
}
