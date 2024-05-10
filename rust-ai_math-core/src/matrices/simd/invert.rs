use crate::matrices::simd::simd_matrix::{SimdCell, SimdMatrix};
use float_cmp::ApproxEq;
use std::simd::{LaneCount, SupportedLaneCount};

impl<T: SimdCell + ApproxEq, const N: usize> SimdMatrix<T, N, N>
where
    LaneCount<{ (N * N).next_power_of_two() }>: SupportedLaneCount,
    LaneCount<{ ((N + 1) * 1).next_power_of_two() }>: SupportedLaneCount,
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

    fn lu_decompose(mut matrix: Self, margin: T::Margin) -> Option<(Self, [isize; N + 1])> {
        let mut p: [isize; N + 1] = [0; N + 1];

        for i in 0..=N {
            p[i] = i as isize;
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

    fn lu_invert(lu_decomposed: &Self, lu_p: &[isize; N + 1]) -> Self {
        let mut IA: SimdMatrix<T, N, N> = SimdMatrix::default();

        for j in 0..N {
            for i in 0..N {
                IA[(i, j)] = if lu_p[i] == j as isize {
                    T::one()
                } else {
                    T::zero()
                };

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

    fn lu_determinant(lu_decomposed: &Self, lu_p: &[isize; N + 1]) -> T {
        let mut det = lu_decomposed[(0, 0)];

        for i in 1..N {
            det = det * lu_decomposed[(i, i)];
        }

        if (lu_p[N] - N as isize) % 2 == 0 {
            det
        } else {
            T::zero() - det
        }
    }
}
