use crate::matrices::simd::{SimdCell, SimdMatrix};
use crate::meta::predicate::{Predicate, Satisfied};
use float_cmp::{ApproxEq, F64Margin};
use num_traits::{ConstOne, ConstZero};
use quickcheck::{empty_shrinker, Arbitrary, Gen};
use std::fmt::{Display, Formatter};
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::simd::{LaneCount, SupportedLaneCount};

/// Can be cast into a square `SimdMatrix` and provides an `Arbitrary` implementation that yields
/// invertible matricies.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct InvertibleTypedMatrix<T: SimdCell, const N: usize>
where
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    matrix: SimdMatrix<T, N, N>,
}

impl<T: SimdCell, R: SimdCell, const N: usize> From<InvertibleTypedMatrix<T, N>>
    for SimdMatrix<R, N, N>
where
    R: From<T>,
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn from(value: InvertibleTypedMatrix<T, N>) -> Self {
        value.matrix.map(|&c| c.into())
    }
}

impl<T: SimdCell, const N: usize> Display for InvertibleTypedMatrix<T, N>
where
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.matrix.fmt(f)
    }
}

impl<T: SimdCell, const N: usize> InvertibleTypedMatrix<T, N>
where
    T: Arbitrary + From<f32> + Into<f64>,
    Predicate<{ N > 1 }>: Satisfied,
    [(); N + 1]:,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
    LaneCount<{ ((N + 1) * 1).next_power_of_two() }>: SupportedLaneCount,
{
    /// We are looking for matricies that can be inverted and multiplied into identity with high precision.
    fn is_candidate(candidate: SimdMatrix<T, N, N>) -> bool {
        let candidate: SimdMatrix<f64, N, N> = candidate.map(|&c| c.into());
        let inverse: Option<(f64, SimdMatrix<f64, N, N>)> = candidate.inverse(F64Margin::default());
        if inverse.is_none() {
            return false;
        }

        let multiplied = candidate * inverse.unwrap().1;

        if !multiplied.is_close_to(SimdMatrix::identity(), F64Margin::default()) {
            return false;
        }

        if candidate.is_close_to(SimdMatrix::identity(), F64Margin::default()) {
            return false;
        }

        true
    }
}

impl<T: SimdCell, const N: usize> Arbitrary for InvertibleTypedMatrix<T, N>
where
    T: Arbitrary + From<f32> + Into<f64>,
    Predicate<{ N > 1 }>: Satisfied,
    [(); N + 1]:,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
    LaneCount<{ ((N + 1) * 1).next_power_of_two() }>: SupportedLaneCount,
{
    fn arbitrary(g: &mut Gen) -> Self {
        loop {
            // An arbitrary multiplicative chain of elementary matricies yields an invertible matrix.
            // We ensure that the invertible matrix can be inverted with high precision, i.e. yields
            // the identity matrix with strong floating point boundaries and is sufficiently distant
            // from the identity matrix.

            let matricies = Vec::<ElementaryTypedMatrix<N>>::arbitrary(g);
            let mut candidate: SimdMatrix<T, N, N> = SimdMatrix::identity();

            for matrix in matricies {
                let m: SimdMatrix<T, N, N> = matrix.into();
                let next_candidate = candidate * m;
                if !Self::is_candidate(next_candidate) {
                    break;
                }

                candidate = next_candidate;
            }

            if Self::is_candidate(candidate) {
                return Self { matrix: candidate };
            }
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

impl<const N: usize> Display for ElementaryTypedMatrix<N>
where
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.matrix.fmt(f)
    }
}

/// Can be cast into a square `SimdMatrix` and provides an `Arbitrary` implementation that yields
/// elementary matricies.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ElementaryTypedMatrix<const N: usize>
where
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    matrix: SimdMatrix<f32, N, N>,
}

impl<R: SimdCell, const N: usize> From<ElementaryTypedMatrix<N>> for SimdMatrix<R, N, N>
where
    R: From<f32>,
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn from(value: ElementaryTypedMatrix<N>) -> Self {
        value.matrix.map(|&c| c.into())
    }
}

impl<const N: usize> ElementaryTypedMatrix<N>
where
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn row_switch_matrix(g: &mut Gen) -> SimdMatrix<f32, N, N> {
        let mut matrix: SimdMatrix<f32, N, N> = SimdMatrix::identity();
        loop {
            let i = usize::arbitrary(g).rem(N);
            let j = usize::arbitrary(g).rem(N);

            if i != j {
                matrix[(i, i)] = 0.0;
                matrix[(j, j)] = 0.0;
                matrix[(i, j)] = 1.0;
                matrix[(j, i)] = 1.0;

                return matrix;
            }
        }
    }

    fn row_mul_matrix(g: &mut Gen) -> SimdMatrix<f32, N, N> {
        let mut matrix: SimdMatrix<f32, N, N> = SimdMatrix::identity();
        loop {
            let i = usize::arbitrary(g).rem(N);
            let value: f32 = Self::arbitrary_cell(g);

            if value != 1.0 && value != 0.0 {
                matrix[(i, i)] = value;
                return matrix;
            }
        }
    }

    fn row_add_matrix(g: &mut Gen) -> SimdMatrix<f32, N, N> {
        let mut matrix = SimdMatrix::identity();
        loop {
            let i = usize::arbitrary(g).rem(N);
            let j = usize::arbitrary(g).rem(N);
            let value = Self::arbitrary_cell(g);

            if i != j && value != 0.0 {
                matrix[(i, j)] = value;
                return matrix;
            }
        }
    }
    fn arbitrary_cell(g: &mut Gen) -> f32 {
        if bool::arbitrary(g) {
            u8::arbitrary(g).rem(5) as f32
        } else {
            -(u8::arbitrary(g).rem(5) as f32)
        }
    }
}

impl<const N: usize> Arbitrary for ElementaryTypedMatrix<N>
where
    Predicate<{ N > 1 }>: Satisfied,
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
{
    fn arbitrary(g: &mut Gen) -> Self {
        Self {
            matrix: match g.choose(&[1.0, 2.0, 3.0]).unwrap() {
                1.0 => Self::row_add_matrix(g),
                2.0 => Self::row_mul_matrix(g),
                _ => Self::row_switch_matrix(g),
            },
        }
    }

    fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
        empty_shrinker()
    }
}

#[cfg(test)]
mod tests {
    use crate::matrices::aliases::Matrix4x4;
    use crate::matrices::typed_matrix_arbitraries::{ElementaryTypedMatrix, InvertibleTypedMatrix};
    use float_cmp::{approx_eq, F32Margin, F64Margin};
    use quickcheck_macros::quickcheck;

    #[quickcheck]
    fn test_invertible_matrix(matrix: InvertibleTypedMatrix<f64, 4>) {
        let x: f64 = 0u8.into();
        let m: Matrix4x4<f64> = matrix.into();
        let (det, inverse) = m.inverse(F64Margin::default()).unwrap();

        assert!(!approx_eq!(Matrix4x4<f64>, m, Matrix4x4::identity()));
        assert!(
            approx_eq!(Matrix4x4<f64>, m * inverse, Matrix4x4::identity()),
            "Matrix [{}] times [{}] did not yield the identity matrix, but [{}].",
            m,
            inverse,
            m * inverse
        );
    }

    #[quickcheck]
    fn test_elementary_matrix(matrix: ElementaryTypedMatrix<4>) {
        let x: f32 = 0u8.into();
        let m: Matrix4x4<f32> = matrix.into();
        let (det, inverse) = m.inverse(F32Margin::default()).unwrap();

        assert!(!approx_eq!(Matrix4x4<f32>, m, Matrix4x4::identity()));
        assert!(
            approx_eq!(Matrix4x4<f32>, m * inverse, Matrix4x4::identity()),
            "Matrix [{}] times [{}] did not yield the identity matrix, but [{}].",
            m,
            inverse,
            m * inverse
        );
    }
}
