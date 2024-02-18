use crate::matrices::vector::Vector;
use crate::meta::predicate::{Predicate, Satisfied};
use num_traits::ConstOne;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

// Common 3D graphics types
pub type Scalar<T> = TypedMatrix<T, 1, 1>;
pub type Vector2<T> = TypedMatrix<T, 2, 1>;
pub type Vector2Transposed<T> = TypedMatrix<T, 1, 2>;
pub type Vector3<T> = TypedMatrix<T, 3, 1>;
pub type Vector3Transposed<T> = TypedMatrix<T, 1, 3>;
pub type Vector4<T> = TypedMatrix<T, 4, 1>;
pub type Vector4Transposed<T> = TypedMatrix<T, 1, 4>;
pub type Matrix2x2<T> = TypedMatrix<T, 2, 2>;
pub type Matrix2x3<T> = TypedMatrix<T, 2, 3>;
pub type Matrix3x2<T> = TypedMatrix<T, 3, 2>;
pub type Matrix3x3<T> = TypedMatrix<T, 3, 3>;
pub type Matrix4x4<T> = TypedMatrix<T, 4, 4>;
pub type Matrix3x4<T> = TypedMatrix<T, 3, 4>;
pub type Matrix4x3<T> = TypedMatrix<T, 3, 4>;

/// This type of matrix is allocated on the stack in column major order with a compile time size.
/// As such it offers the fastest possible speed for CPU execution and is well suited to represent
/// types commonly used in 3D engines.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TypedMatrix<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> {
    rows: [[T; ROW_COUNT]; COLUMN_COUNT],
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> TypedMatrix<T, COLUMN_COUNT, ROW_COUNT> {
    fn row_count(&self) -> usize {
        ROW_COUNT
    }

    fn column_count(&self) -> usize {
        COLUMN_COUNT
    }

    fn element_count(&self) -> usize {
        ROW_COUNT * COLUMN_COUNT
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> TypedMatrix<T, COLUMN_COUNT, ROW_COUNT>
where
    T: Copy + Default,
{
    pub fn transposed(&self) -> TypedMatrix<T, ROW_COUNT, COLUMN_COUNT> {
        let mut transposed = TypedMatrix::<T, ROW_COUNT, COLUMN_COUNT>::default();
        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                transposed[(j, i)] = self[(i, j)];
            }
        }

        transposed
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Default + ConstOne,
{
    pub fn identity() -> TypedMatrix<T, ROW_COUNT, COLUMN_COUNT> {
        let mut result = Self::default();
        for i in 0..ROW_COUNT.min(COLUMN_COUNT) {
            result[(i, i)] = T::ONE;
        }
        result
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Default,
{
    pub fn inverse() -> TypedMatrix<T, ROW_COUNT, COLUMN_COUNT> {
        let mut result = Self::default();
        for i in 0..ROW_COUNT.min(COLUMN_COUNT) {}
        result
    }
}

impl<
        T,
        const LEFT_ROW_COUNT: usize,
        const COMMON_COUNT: usize,
        const RIGHT_COLUMN_COUNT: usize,
    > Mul<TypedMatrix<T, COMMON_COUNT, RIGHT_COLUMN_COUNT>>
    for TypedMatrix<T, LEFT_ROW_COUNT, COMMON_COUNT>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    type Output = TypedMatrix<T, LEFT_ROW_COUNT, RIGHT_COLUMN_COUNT>;

    fn mul(self, rhs: TypedMatrix<T, COMMON_COUNT, RIGHT_COLUMN_COUNT>) -> Self::Output {
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

impl<T, const N: usize> MulAssign<TypedMatrix<T, N, N>> for TypedMatrix<T, N, N>
where
    T: Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T, const ROW_COUNT: usize> Vector for TypedMatrix<T, ROW_COUNT, 1>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
    Predicate<{ ROW_COUNT > 1 }>: Satisfied,
{
    type Cell = T;

    fn dimension(&self) -> usize {
        ROW_COUNT
    }
}

impl<T, const COLUMN_COUNT: usize> Vector for TypedMatrix<T, 1, COLUMN_COUNT>
where
    T: Copy + Eq + Sub<Output = T> + Add<Output = T> + Debug + Default,
    Predicate<{ COLUMN_COUNT > 1 }>: Satisfied,
{
    type Cell = T;

    fn dimension(&self) -> usize {
        COLUMN_COUNT
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Index<(usize, usize)>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.rows[index.1][index.0]
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> IndexMut<(usize, usize)>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.rows[index.1][index.0]
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Index<usize>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index / ROW_COUNT][index % ROW_COUNT]
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> IndexMut<usize>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index / ROW_COUNT][index % ROW_COUNT]
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    AddAssign<TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>> for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Add<Output = T> + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Add
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] + other.rows[i][j]
            }
        }

        result
    }
}

impl<T, TRhs, const N: usize> MulAssign<TRhs> for TypedMatrix<T, N, N>
where
    T: Mul<TRhs, Output = T> + Copy,
{
    fn mul_assign(&mut self, rhs: TRhs) {
        *self = *self * rhs;
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Mul<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] * other
            }
        }

        result
    }
}

impl<T, const N: usize> AddAssign<T> for TypedMatrix<T, N, N>
where
    T: Add<Output = T> + Copy,
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Add<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Add<Output = T>,
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] + other
            }
        }

        result
    }
}

impl<T, const N: usize> DivAssign<T> for TypedMatrix<T, N, N>
where
    T: Div<Output = T> + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T, const N: usize> SubAssign<T> for TypedMatrix<T, N, N>
where
    T: Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Sub<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] - other
            }
        }

        result
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Div<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.rows[i][j] = self.rows[i][j] / other
            }
        }

        result
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    SubAssign<TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>> for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Sub<Output = T> + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Sub
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
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

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Default
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self {
            rows: [[Default::default(); ROW_COUNT]; COLUMN_COUNT],
        }
    }
}

impl<T, const ROW_COUNT: usize, const COLUMN_COUNT: usize> From<[[T; ROW_COUNT]; COLUMN_COUNT]>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn from(value: [[T; ROW_COUNT]; COLUMN_COUNT]) -> Self {
        Self { rows: value }
    }
}

impl<T, const ROW_COUNT: usize> From<[T; ROW_COUNT]> for TypedMatrix<T, ROW_COUNT, 1> {
    fn from(value: [T; ROW_COUNT]) -> Self {
        Self { rows: [value] }
    }
}

impl<T> From<T> for TypedMatrix<T, 1, 1> {
    fn from(value: T) -> Self {
        Self { rows: [[value]] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::typed_matrix::TypedMatrix;

    #[test]
    fn test_vector2() {
        let vec2: Vector2<u8> = [1, 2].into();
        assert_eq!(vec2.row_count(), 2);
        assert_eq!(vec2.column_count(), 1);
        assert_eq!(vec2[(0, 0)], 1);
        assert_eq!(vec2[(1, 0)], 2);
        assert_eq!(vec2[0], 1);
        assert_eq!(vec2[1], 2);
    }

    #[test]
    fn test_vector2_transposed() {
        let vec2: Vector2Transposed<u8> = [[1], [2]].into();
        assert_eq!(vec2.row_count(), 1);
        assert_eq!(vec2.column_count(), 2);
        assert_eq!(vec2[(0, 0)], 1);
        assert_eq!(vec2[(0, 1)], 2);
        assert_eq!(vec2[0], 1);
        assert_eq!(vec2[1], 2);

        let vec2_transposed: Vector2<u8> = vec2.transposed();
        assert_eq!(vec2_transposed, [1, 2].into());
    }

    #[test]
    fn test_vector3() {
        let vec3: Vector3<u8> = [1, 2, 3].into();
        assert_eq!(vec3.row_count(), 3);
        assert_eq!(vec3.column_count(), 1);
        assert_eq!(vec3[(0, 0)], 1);
        assert_eq!(vec3[(1, 0)], 2);
        assert_eq!(vec3[(2, 0)], 3);
        assert_eq!(vec3[0], 1);
        assert_eq!(vec3[1], 2);
        assert_eq!(vec3[2], 3);
    }

    #[test]
    fn test_vector3_transposed() {
        let vec3: Vector3Transposed<u8> = [[1], [2], [3]].into();
        assert_eq!(vec3.row_count(), 1);
        assert_eq!(vec3.column_count(), 3);
        assert_eq!(vec3[(0, 0)], 1);
        assert_eq!(vec3[(0, 1)], 2);
        assert_eq!(vec3[(0, 2)], 3);
        assert_eq!(vec3[0], 1);
        assert_eq!(vec3[1], 2);
        assert_eq!(vec3[2], 3);

        let vec3_transposed: Vector3<u8> = vec3.transposed();
        assert_eq!(vec3_transposed, [1, 2, 3].into());
    }

    #[test]
    fn test_vector4() {
        let vec4: Vector4<u8> = [1, 2, 3, 4].into();
        assert_eq!(vec4.row_count(), 4);
        assert_eq!(vec4.column_count(), 1);
        assert_eq!(vec4[(0, 0)], 1);
        assert_eq!(vec4[(1, 0)], 2);
        assert_eq!(vec4[(2, 0)], 3);
        assert_eq!(vec4[(3, 0)], 4);
        assert_eq!(vec4[0], 1);
        assert_eq!(vec4[1], 2);
        assert_eq!(vec4[2], 3);
        assert_eq!(vec4[3], 4);
    }

    #[test]
    fn test_vector4_transposed() {
        let vec4: Vector4Transposed<u8> = [[1], [2], [3], [4]].into();
        assert_eq!(vec4.row_count(), 1);
        assert_eq!(vec4.column_count(), 4);
        assert_eq!(vec4[(0, 0)], 1);
        assert_eq!(vec4[(0, 1)], 2);
        assert_eq!(vec4[(0, 2)], 3);
        assert_eq!(vec4[(0, 3)], 4);
        assert_eq!(vec4[0], 1);
        assert_eq!(vec4[1], 2);
        assert_eq!(vec4[2], 3);
        assert_eq!(vec4[3], 4);

        let vec4_transposed: Vector4<u8> = vec4.transposed();
        assert_eq!(vec4_transposed, [1, 2, 3, 4].into());
    }

    #[test]
    fn test_matrix_product() {
        let vec3: Vector3<u8> = [1, 2, 3].into();
        let vec3_transposed: Vector3Transposed<u8> = [[1], [2], [3]].into();

        let product: Matrix3x3<u8> = vec3 * vec3_transposed;
        assert_eq!(product, [[1, 2, 3], [2, 4, 6], [3, 6, 9]].into());

        let product: Scalar<u8> = vec3_transposed * vec3;
        assert_eq!(product[0], 14);

        let mut product: Matrix2x2<u8> = [[1, 2], [2, 3]].into();
        let rhs: Matrix2x2<u8> = [[3, 4], [5, 6]].into();
        product *= rhs;
        assert_eq!(product, [[11, 18], [17, 28]].into());
    }

    #[test]
    fn test_matrix_scalars() {
        let mut vec3: Vector3<u8> = [1, 2, 3].into();
        vec3 *= 2;
    }

    #[test]
    fn test_matrix_identity() {
        assert_eq!(
            Matrix3x3::<u8>::identity() * Matrix3x3::<u8>::identity(),
            Matrix3x3::<u8>::identity()
        );

        assert_eq!(
            Matrix3x3::<u8>::identity(),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]].into()
        );

        assert_eq!(Matrix2x3::<u8>::identity(), [[1, 0], [0, 1], [0, 0]].into());
    }
}
