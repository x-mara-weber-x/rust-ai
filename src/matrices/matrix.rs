use crate::matrices::typed_matrix::{CellBase, CellFloat, CellInteger, TypedMatrix};
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Index, IndexMut};

pub trait Matrix<T: CellBase>:
    Debug
    + Clone
    + Display
    + Index<(usize, usize), Output = T>
    + IndexMut<(usize, usize)>
    + Index<usize>
    + IndexMut<usize>
{
    type Transposed: Matrix<T>;
    type Mapped<R: CellBase>: Matrix<R>;

    fn new(row_count: usize, column_count: usize) -> Self;

    fn row_count(&self) -> usize;

    fn column_count(&self) -> usize;

    fn element_count(&self) -> usize;

    fn map<F, R: CellInteger>(&self, mapper: F) -> Self::Mapped<R>
    where
        F: Fn(&T) -> R,
    {
        let mut result = Self::Mapped::<R>::new(self.row_count(), self.column_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                result[(i, j)] = mapper(&self[(i, j)]);
            }
        }

        result
    }

    fn swap_row(&mut self, row_a: usize, row_b: usize) {
        for col in 0..self.column_count() {
            (self[(row_a, col)], self[(row_b, col)]) = (self[(row_b, col)], self[(row_a, col)]);
        }
    }

    fn transposed(&self) -> Self::Transposed {
        let mut transposed = Self::Transposed::new(self.column_count(), self.row_count());

        for i in 0..self.row_count() {
            for j in 0..self.column_count() {
                transposed[(j, i)] = self[(i, j)];
            }
        }

        transposed
    }

    fn is_close_to<M: Into<T::Margin> + Clone>(&self, other: Self, margin: M) -> bool
    where
        T: CellFloat,
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
