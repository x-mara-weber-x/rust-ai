use crate::matrices::matrix::Matrix;
use crate::matrices::stack_matrix::StackMatrix;
use std::fmt::Debug;
use std::ops::{Add, Index, IndexMut, Sub};

#[derive(Debug, Clone)]
pub struct HeapMatrix<T> {
    cells: Vec<T>,
    row_count: usize,
    column_count: usize,
}

impl<T> HeapMatrix<T>
where
    T: Copy + Debug + Default,
{
    fn new(row_count: usize, column_count: usize) -> Self {
        Self {
            cells: vec![Default::default(); row_count * column_count],
            row_count,
            column_count,
        }
    }
}

impl<T> Matrix for HeapMatrix<T>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
{
    type Transposed = Self;
    type Cell = T;

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn column_count(&self) -> usize {
        self.column_count
    }

    fn transposed(&self) -> Self::Transposed {
        let mut transposed = Self::new(self.column_count, self.row_count);
        for i in 0..self.row_count {
            for j in 0..self.column_count {
                transposed[(i, j)] = self[(j, i)];
            }
        }

        transposed
    }

    fn element_count(&self) -> usize {
        self.cells.len()
    }
}

impl<T> Index<(usize, usize)> for HeapMatrix<T>
where
    T: Copy,
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.cells[(index.0 * self.row_count) + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for HeapMatrix<T>
where
    T: Copy,
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.cells[(index.0 * self.row_count) + index.1]
    }
}

impl<T> Index<usize> for HeapMatrix<T>
where
    T: Copy,
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cells[index]
    }
}

impl<T> IndexMut<usize> for HeapMatrix<T>
where
    T: Copy,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cells[index]
    }
}

impl<T> Default for HeapMatrix<T>
where
    T: Copy,
{
    fn default() -> Self {
        Self {
            cells: Default::default(),
            row_count: 0,
            column_count: 0,
        }
    }
}

impl<T> Eq for HeapMatrix<T> where T: Copy + Eq {}

impl<T> PartialEq for HeapMatrix<T>
where
    T: Copy + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        self.cells == other.cells
    }
}

impl<T> Add for HeapMatrix<T>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self.clone();

        for i in 0..self.cells.len() {
            result.cells[i] = self.cells[i] + other.cells[i]
        }

        result
    }
}

impl<T> Sub for HeapMatrix<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self.clone();

        for i in 0..self.cells.len() {
            result.cells[i] = self.cells[i] - other.cells[i]
        }

        result
    }
}
