use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

pub trait Vector:
    Add<Output = Self>
    + Sub<Output = Self>
    + Sized
    + Default
    + Clone
    + Eq
    + Debug
    + Index<usize, Output = Self::Cell>
    + IndexMut<usize, Output = Self::Cell>
{
    type Cell: Copy
        + Eq
        + Default
        + Add<Output = Self::Cell>
        //        + AddAssign
        + Sub<Output = Self::Cell>;

    fn dimension(&self) -> usize;

    fn distance(&self, other: &Self) -> f64 {
        // let h = self.sub(*other);
        // (*self - *other).magnitude()
        0.0
    }

    fn magnitude(&self) -> f64 {
        // let sqr: f64 = self.sqrMagnitude().into();
        // sqr.sqrt()
        0.0
    }

    fn sqrMagnitude(&self) -> Self::Cell {
        let mut sqrMagnitude = Default::default();
        // for i in 0..self.dimension() {
        //     sqrMagnitude += self[i] * self[i];
        // }
        sqrMagnitude
    }
}
