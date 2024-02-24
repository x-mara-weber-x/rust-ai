use crate::matrices::typed_matrix::{CellBase, CellFloat, CellInteger, Vector3};
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub trait Vector<T: CellInteger>:
    CellBase
    + Index<usize, Output = T>
    + IndexMut<usize, Output = T>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Div<T, Output = Self>
    + Mul<T, Output = Self>
    + Default
{
    fn dimension(&self) -> usize;

    fn new(values: &[T]) -> Self {
        let mut result = Self::zero();
        for i in 0..result.dimension().min(values.len()) {
            result[i] = values[i];
        }
        result
    }

    fn zero() -> Self {
        Self::default()
    }

    fn div(self, rhs: T) -> Self {
        let mut result = self;
        for i in 0..result.dimension() {
            result[i] /= rhs;
        }
        result
    }

    fn mul(self, rhs: T) -> Self {
        let mut result = self;
        for i in 0..result.dimension() {
            result[i] *= rhs;
        }
        result
    }

    fn one() -> Self {
        let mut result = Self::default();
        for i in 0..result.dimension() {
            result[i] = T::one();
        }
        result
    }

    fn distance(&self, other: &Self) -> T
    where
        T: CellFloat,
    {
        (*self - *other).magnitude()
    }

    fn normalized(&self) -> Self
    where
        T: CellFloat,
    {
        *self / self.magnitude()
    }

    fn magnitude(&self) -> T
    where
        T: CellFloat,
    {
        self.sqr_magnitude().sqrt()
    }

    fn sqr_magnitude(&self) -> T
    where
        T: CellFloat,
    {
        self.dot_product(self)
    }

    fn require_same_dimension(&self, other: &Self) {
        if self.dimension() != other.dimension() {
            panic!("Attempt to operate on two vectors with different dimension.");
        }
    }

    fn dot_product(&self, other: &Self) -> T
    where
        T: CellFloat,
    {
        self.require_same_dimension(other);

        let mut result = Default::default();
        for i in 0..self.dimension() {
            result += self[i] * other[i];
        }
        result
    }

    fn angle(&self, other: &Self) -> T
    where
        T: CellFloat,
    {
        (self.dot_product(other) / (self.magnitude() * other.magnitude())).acos()
    }
}
