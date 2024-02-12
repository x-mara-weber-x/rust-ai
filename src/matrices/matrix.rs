use crate::matrices::stack_matrix::StackMatrix;
use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

pub trait Matrix:
    Sub<Output = Self>
    + Sized
    + Default
    + Clone
    + Eq
    + Debug
    //+ IntoIterator
    + Index<usize>
    + IndexMut<usize>
    + Index<(usize, usize)>
    + IndexMut<(usize, usize)>
{
    type Cell;
    type Transposed: Matrix;

    fn row_count(&self) -> usize;

    fn column_count(&self) -> usize;

    fn element_count(&self) -> usize;

    fn transposed(&self) -> Self::Transposed;

    fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();

        result
    }

    // fn iter(&self) -> Iter<Self> {
    //     Iter::new(self)
    // }
}

pub struct Iter<TMatrix: Matrix> {
    matrix: TMatrix,
    index: usize,
}

impl<TMatrix: Matrix> Iter<TMatrix> {
    fn new(matrix: TMatrix) -> Self {
        Self { index: 0, matrix }
    }
}

// impl<TMatrix: Matrix> Iterator for Iter<TMatrix> {
//     type Item = TMatrix::Cell;
//
//     fn next(&mut self) -> Option<Self::Item> {
//         if self.index >= self.matrix.element_count() {
//             return None;
//         } else {
//             self.index += 1;
//             self.matrix[self.index - 1]
//         }
//     }
// }
