use crate::matrices::matrix::Matrix;
use crate::matrices::vector::Vector;
use crate::meta::predicate::{Predicate, Satisfied};
use float_cmp::ApproxEq;
use num_traits::real::Real;
use num_traits::{NumAssign, NumAssignRef, One, Zero};
use quickcheck::{empty_shrinker, Arbitrary, Gen};
use std::any::type_name;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
};

// Common 3D graphics types
pub type Scalar<T> = TypedVector<T, 1>;
pub type Vector2<T> = TypedVector<T, 2>;
pub type Vector2Transposed<T> = TypedVectorTransposed<T, 2>;
pub type Vector3<T> = TypedVector<T, 3>;
pub type Vector3Transposed<T> = TypedVectorTransposed<T, 3>;
pub type Vector4<T> = TypedVector<T, 4>;
pub type Vector4Transposed<T> = TypedVectorTransposed<T, 4>;
pub type Matrix2x2<T> = TypedMatrix<T, 2, 2>;
pub type Matrix2x3<T> = TypedMatrix<T, 2, 3>;
pub type Matrix3x2<T> = TypedMatrix<T, 3, 2>;
pub type Matrix3x3<T> = TypedMatrix<T, 3, 3>;
pub type Matrix4x4<T> = TypedMatrix<T, 4, 4>;
pub type Matrix3x4<T> = TypedMatrix<T, 3, 4>;
pub type Matrix4x3<T> = TypedMatrix<T, 3, 4>;
pub type TypedVector<T, const DIMENSION: usize> = TypedMatrix<T, DIMENSION, 1>;
pub type TypedVectorTransposed<T, const DIMENSION: usize> = TypedMatrix<T, 1, DIMENSION>;

/// This cell type provides the basics for matrix operations that don't need any concept of numbers
pub trait CellBase: 'static + Debug + Default + Copy + Clone + Display {}
impl<T> CellBase for T where T: 'static + Debug + Default + Copy + Clone + Display {}

/// This cell type is needed for most matrix operations and is compatible with Rust's integers.
pub trait CellInteger: CellBase + NumAssignRef + Arbitrary {}

impl<T> CellInteger for T where
    T: CellBase
        + NumAssignRef
        + Mul<Output = Self>
        + MulAssign
        + Add<Output = Self>
        + AddAssign
        + Div<Output = Self>
        + DivAssign
        + Sub<Output = Self>
        + SubAssign
        + RemAssign
        + Rem<Output = Self>
        + PartialEq
        + PartialOrd
        + One
        + Zero
        + Arbitrary
{
}

/// This cell type is needed for most advanced matrix operations and is compatible with Rust's floats.
pub trait CellFloat: CellInteger + Real + ApproxEq {}

impl<T> CellFloat for T where T: CellInteger + Real + ApproxEq {}

/// This type of matrix is allocated on the stack in column major order with a compile time size.
/// As such it offers the fastest possible speed for CPU execution and is well suited to represent
/// types commonly used in 3D engines.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct TypedMatrix<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> {
    cols: [[T; ROW_COUNT]; COLUMN_COUNT],
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Display
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Matrix::fmt(self, f)
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Index<(usize, usize)>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        &self.cols[index.1][index.0]
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> IndexMut<(usize, usize)>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        &mut self.cols[index.1][index.0]
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Matrix<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Transposed = TypedMatrix<T, COLUMN_COUNT, ROW_COUNT>;

    fn new(row_count: usize, column_count: usize) -> Self {
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

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    pub fn identity() -> TypedMatrix<T, ROW_COUNT, COLUMN_COUNT> {
        let mut result = Self::default();
        for i in 0..ROW_COUNT.min(COLUMN_COUNT) {
            result[(i, i)] = T::one();
        }
        result
    }
}

impl<T: CellInteger, const ROW_COUNT: usize> Vector<T> for TypedMatrix<T, ROW_COUNT, 1>
where
    Predicate<{ ROW_COUNT > 1 }>: Satisfied,
{
    fn dimension(&self) -> usize {
        ROW_COUNT
    }
}

impl<T: CellInteger, const COLUMN_COUNT: usize> Vector<T> for TypedMatrix<T, 1, COLUMN_COUNT>
where
    Predicate<{ COLUMN_COUNT > 1 }>: Satisfied,
{
    fn dimension(&self) -> usize {
        COLUMN_COUNT
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Index<usize>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.cols[index / ROW_COUNT][index % ROW_COUNT]
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> IndexMut<usize>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cols[index / ROW_COUNT][index % ROW_COUNT]
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    AddAssign<TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>> for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Add
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.cols[i][j] = self.cols[i][j] + other.cols[i][j]
            }
        }

        result
    }
}

impl<T: CellInteger, const N: usize> MulAssign<TypedMatrix<T, N, N>> for TypedMatrix<T, N, N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<
        T: CellInteger,
        const LEFT_ROW_COUNT: usize,
        const COMMON_COUNT: usize,
        const RIGHT_COLUMN_COUNT: usize,
    > Mul<TypedMatrix<T, COMMON_COUNT, RIGHT_COLUMN_COUNT>>
    for TypedMatrix<T, LEFT_ROW_COUNT, COMMON_COUNT>
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

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> MulAssign<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Mul<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = Self;

    fn mul(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.cols[i][j] = self.cols[i][j] * other
            }
        }

        result
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> AddAssign<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs;
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Add<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = Self;

    fn add(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.cols[i][j] = self.cols[i][j] + other
            }
        }

        result
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> DivAssign<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> SubAssign<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs;
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Sub<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = Self;

    fn sub(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.cols[i][j] = self.cols[i][j] - other
            }
        }

        result
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Div<T>
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = Self;

    fn div(self, other: T) -> Self::Output {
        let mut result = self.clone();

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.cols[i][j] = self.cols[i][j] / other
            }
        }

        result
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    SubAssign<TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>> for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<T: CellInteger, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Sub
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self;

        for i in 0..COLUMN_COUNT {
            for j in 0..ROW_COUNT {
                result.cols[i][j] = self.cols[i][j] - other.cols[i][j]
            }
        }

        result
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Default
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn default() -> Self {
        Self {
            cols: [[Default::default(); ROW_COUNT]; COLUMN_COUNT],
        }
    }
}

impl<T: CellFloat, const ROW_COUNT: usize, const COLUMN_COUNT: usize> ApproxEq
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
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

impl<T: CellBase + Arbitrary, const ROW_COUNT: usize, const COLUMN_COUNT: usize> Arbitrary
    for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
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

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

impl<T: CellBase, const ROW_COUNT: usize, const COLUMN_COUNT: usize>
    From<[[T; ROW_COUNT]; COLUMN_COUNT]> for TypedMatrix<T, ROW_COUNT, COLUMN_COUNT>
{
    fn from(value: [[T; ROW_COUNT]; COLUMN_COUNT]) -> Self {
        Self { cols: value }
    }
}

impl<T: CellBase, const ROW_COUNT: usize> From<[T; ROW_COUNT]> for TypedMatrix<T, ROW_COUNT, 1> {
    fn from(value: [T; ROW_COUNT]) -> Self {
        Self { cols: [value] }
    }
}

impl<T: CellBase> From<T> for TypedMatrix<T, 1, 1> {
    fn from(value: T) -> Self {
        Self { cols: [[value]] }
    }
}

impl<T: CellFloat, const N: usize> TypedMatrix<T, N, N>
where
    [(); N + 1]:,
{
    pub fn inverse(&self, margin: T::Margin) -> Option<(T, Self)> {
        Self::lu_decompose(*self, margin).map(|lu| {
            (
                Self::lu_determinant(&lu.0, &lu.1),
                Self::lu_invert(&lu.0, &lu.1),
            )
        })
    }

    fn lu_decompose(
        mut matrix: Self,
        margin: T::Margin,
    ) -> Option<(Self, TypedMatrix<usize, { N + 1 }, 1>)> {
        let mut p: TypedMatrix<usize, { N + 1 }, 1> = TypedMatrix::default();

        for i in 0..=N {
            p[i] = i;
        }

        for i in 0..N {
            let mut max_a: T = T::zero();
            let mut imax = i;

            for k in i..N {
                let mut abs_a: T = matrix[(k, i)];
                if abs_a < T::zero() {
                    abs_a = T::zero() - abs_a;
                }

                if abs_a > max_a {
                    max_a = abs_a;
                    imax = k;
                }
            }

            if max_a.approx_eq(T::zero(), margin) {
                // matrix is degenerate
                return None;
            }

            if imax != i {
                (p[i], p[imax]) = (p[imax], p[i]);
                matrix.swap_row(i, imax);

                // counting pivots starting from N (for determinant)
                p[N] = p[N] + 1;
            }

            for j in i + 1..N {
                matrix[(j, i)] = matrix[(j, i)] / matrix[(i, i)];

                for k in i + 1..N {
                    matrix[(j, k)] = matrix[(j, k)] - matrix[(j, i)] * matrix[(i, k)];
                }
            }
        }

        Some((matrix, p))
    }

    fn lu_invert(lu_decomposed: &Self, lu_p: &TypedMatrix<usize, { N + 1 }, 1>) -> Self {
        let mut IA: TypedMatrix<T, N, N> = TypedMatrix::default();

        for j in 0..N {
            for i in 0..N {
                IA[(i, j)] = if lu_p[i] == j { T::one() } else { T::zero() };

                for k in 0..i {
                    IA[(i, j)] = IA[(i, j)] - lu_decomposed[(i, k)] * IA[(k, j)];
                }
            }

            for i in (0..=N - 1).rev() {
                for k in i + 1..N {
                    IA[(i, j)] = IA[(i, j)] - lu_decomposed[(i, k)] * IA[(k, j)];
                }

                IA[(i, j)] = IA[(i, j)] / lu_decomposed[(i, i)];
            }
        }

        IA
    }

    fn lu_determinant(lu_decomposed: &Self, lu_p: &TypedMatrix<usize, { N + 1 }, 1>) -> T {
        let mut det = lu_decomposed[(0, 0)];

        for i in 1..N {
            det = det * lu_decomposed[(i, i)];
        }

        if (lu_p[N] - N) % 2 == 0 {
            det
        } else {
            T::zero() - det
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrices::typed_matrix::TypedMatrix;
    use float_cmp::{approx_eq, F32Margin, F64Margin};

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
    fn test_matrix_addition() {
        let matrixA: Matrix3x2<u8> = [[1, 2, 3], [2, 3, 4]].into();
        let matrixB: Matrix3x2<u8> = [[2, 3, 4], [5, 6, 7]].into();
        assert_eq!(matrixA + matrixB, [[3, 5, 7], [7, 9, 11]].into());

        let mut tmp = matrixA;
        tmp += matrixB;
        assert_eq!(tmp, [[3, 5, 7], [7, 9, 11]].into());
    }

    #[test]
    fn test_matrix_subtraction() {
        let matrixA: Matrix3x2<i8> = [[1, 2, 3], [2, 3, 4]].into();
        let matrixB: Matrix3x2<i8> = [[2, 3, 4], [5, 6, 7]].into();
        assert_eq!(matrixA - matrixB, [[-1, -1, -1], [-3, -3, -3]].into());

        let mut tmp = matrixA;
        tmp -= matrixB;
        assert_eq!(tmp, [[-1, -1, -1], [-3, -3, -3]].into());
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
        // multiplication
        let mut vec3: Vector3<u8> = [1, 2, 3].into();
        vec3 *= 2;
        assert_eq!(vec3, [2, 4, 6].into());
        let vec3: Vector3<u8> = [1, 2, 3].into();
        assert_eq!(vec3 * 2, [2, 4, 6].into());

        // division
        let mut vec3: Vector3<u8> = [1, 2, 3].into();
        vec3 /= 2;
        assert_eq!(vec3, [0, 1, 1].into());
        let vec3: Vector3<u8> = [1, 2, 3].into();
        assert_eq!(vec3 / 2, [0, 1, 1].into());

        // addition
        let mut vec3: Vector3<u8> = [1, 2, 3].into();
        vec3 += 2;
        assert_eq!(vec3, [3, 4, 5].into());
        let vec3: Vector3<u8> = [1, 2, 3].into();
        assert_eq!(vec3 + 2, [3, 4, 5].into());

        // subtraction
        let mut vec3: Vector3<i8> = [1, 2, 3].into();
        vec3 -= 2;
        assert_eq!(vec3, [-1, 0, 1].into());
        let vec3: Vector3<i8> = [1, 2, 3].into();
        assert_eq!(vec3 - 2, [-1, 0, 1].into());

        // conversion
        let scalar: Scalar<u8> = [[1]].into();
        assert_eq!(scalar, 1.into());
        let scalar: Scalar<u8> = [1].into();
        assert_eq!(scalar, 1.into());
        let scalar: Scalar<u8> = 1.into();
        assert_eq!(scalar[0], 1);
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

    use crate::matrices::typed_matrix_arbitraries::InvertibleTypedMatrix;
    use quickcheck_macros::quickcheck;

    #[test]
    fn test_identity() {
        let matrix: Matrix2x2<f32> = [[1.0, 0.0], [0.0, 1.0]].into();
        let (det, inverse) = matrix.inverse(F32Margin::default()).unwrap();

        assert_eq!(inverse, [[1.0, 0.0], [0.0, 1.0]].into());
        assert_eq!(det, 1.0);
    }

    #[test]
    fn test_other() {
        let matrix: Matrix2x2<f32> = [[2.0, 4.0], [1.0, 3.0]].into();
        let (det, inverse) = matrix.inverse(F32Margin::default()).unwrap();

        assert_eq!(inverse, [[1.5, -2.0], [-0.5, 1.0]].into());
        assert_eq!(det, 2.0);
    }

    #[quickcheck]
    fn test(matrix: InvertibleTypedMatrix<f64, 4>) {
        let matrix: Matrix4x4<f64> = matrix.into();
        let (det, inverse) = matrix.inverse(F64Margin::default()).unwrap();

        assert!(
            approx_eq!(Matrix4x4<f64>, matrix * inverse, Matrix4x4::identity()),
            "Matrix [{}] times [{}] did not yield the identity matrix, but [{}].",
            matrix,
            inverse,
            matrix * inverse
        );
    }
}
