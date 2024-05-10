#[cfg(test)]
mod tests {
    use float_cmp::{approx_eq, F32Margin, F64Margin};
    use crate::matrices::aliases::Vector2Transposed;
    use crate::matrices::*;

    #[test]
    fn test_vector2() {
        let vec2: Vector2<i8> = [1, 2].into();
        assert_eq!(vec2.row_count(), 2);
        assert_eq!(vec2.column_count(), 1);
        assert_eq!(vec2[(0, 0)], 1);
        assert_eq!(vec2[(1, 0)], 2);
        assert_eq!(vec2[0], 1);
        assert_eq!(vec2[1], 2);
    }

    #[test]
    fn test_vector2_transposed() {
        let vec2: Vector2Transposed<i8> = [[1], [2]].into();
        assert_eq!(vec2.row_count(), 1);
        assert_eq!(vec2.column_count(), 2);
        assert_eq!(vec2[(0, 0)], 1);
        assert_eq!(vec2[(0, 1)], 2);
        assert_eq!(vec2[0], 1);
        assert_eq!(vec2[1], 2);

        let vec2_transposed: Vector2<i8> = vec2.transposed();
        assert_eq!(vec2_transposed, [1, 2].into());
    }

    #[test]
    fn test_vector3() {
        let vec3: Vector3<i8> = [1, 2, 3].into();
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
        let vec3: Vector3Transposed<i8> = [[1], [2], [3]].into();
        assert_eq!(vec3.row_count(), 1);
        assert_eq!(vec3.column_count(), 3);
        assert_eq!(vec3[(0, 0)], 1);
        assert_eq!(vec3[(0, 1)], 2);
        assert_eq!(vec3[(0, 2)], 3);
        assert_eq!(vec3[0], 1);
        assert_eq!(vec3[1], 2);
        assert_eq!(vec3[2], 3);

        let vec3_transposed: Vector3<i8> = vec3.transposed();
        assert_eq!(vec3_transposed, [1, 2, 3].into());
    }

    #[test]
    fn test_vector4() {
        let vec4: Vector4<i8> = [1, 2, 3, 4].into();
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
        let vec4: Vector4Transposed<i8> = [[1], [2], [3], [4]].into();
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

        let vec4_transposed: Vector4<i8> = vec4.transposed();
        assert_eq!(vec4_transposed, [1, 2, 3, 4].into());
    }

    #[test]
    fn test_matrix_addition() {
        let matrixA: Matrix3x2<i8> = [[1, 2, 3], [2, 3, 4]].into();
        let matrixB: Matrix3x2<i8> = [[2, 3, 4], [5, 6, 7]].into();
        assert_eq!(matrixA + matrixB, [[3, 5, 7], [7, 9, 11]].into());

        let mut tmp = matrixA;
        tmp = tmp + matrixB;
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
        let vec3: Vector3<i8> = [1, 2, 3].into();
        let vec3_transposed: Vector3Transposed<i8> = [[1], [2], [3]].into();

        let product: Matrix3x3<i8> = vec3 * vec3_transposed;
        assert_eq!(product, [[1, 2, 3], [2, 4, 6], [3, 6, 9]].into());

        let product: Scalar<i8> = vec3_transposed * vec3;
        assert_eq!(product[0], 14);

        let mut product: Matrix2x2<i8> = [[1, 2], [2, 3]].into();
        let rhs: Matrix2x2<i8> = [[3, 4], [5, 6]].into();
        product *= rhs;
        assert_eq!(product, [[11, 18], [17, 28]].into());
    }

    #[test]
    fn test_matrix_scalars() {
        // multiplication
        let mut vec3: Vector3<i8> = [1, 2, 3].into();
        vec3 *= 2;
        assert_eq!(vec3, [2, 4, 6].into());
        let vec3: Vector3<i8> = [1, 2, 3].into();
        assert_eq!(vec3 * 2i8, [2, 4, 6].into());

        // division
        let mut vec3: Vector3<i8> = [1, 2, 3].into();
        vec3 /= 2;
        assert_eq!(vec3, [0, 1, 1].into());
        let vec3: Vector3<i8> = [1, 2, 3].into();
        assert_eq!(vec3 / 2, [0, 1, 1].into());

        // addition
        let mut vec3: Vector3<i8> = [1, 2, 3].into();
        vec3 += 2;
        assert_eq!(vec3, [3, 4, 5].into());
        let vec3: Vector3<i8> = [1, 2, 3].into();
        assert_eq!(vec3 + 2, [3, 4, 5].into());

        // subtraction
        let mut vec3: Vector3<i8> = [1, 2, 3].into();
        vec3 -= 2;
        assert_eq!(vec3, [-1, 0, 1].into());
        let vec3: Vector3<i8> = [1, 2, 3].into();
        assert_eq!(vec3 - 2, [-1, 0, 1].into());

        // conversion
        let scalar: Scalar<i8> = [[1]].into();
        assert_eq!(scalar, 1.into());
        let scalar: Scalar<i8> = [1].into();
        assert_eq!(scalar, 1.into());
        let scalar: Scalar<i8> = 1.into();
        assert_eq!(scalar[0], 1);
    }

    #[test]
    fn test_matrix_indexing() {
        let m: SimdMatrix<i8, 3,5> = [
            [1, 2, 3],
            [4,5,6],
            [7,8,9],
            [10,11,12],
            [13,14,15]
        ].into();

        assert_eq!(m[(0,0)], 1);
        assert_eq!(m[(1,0)], 2);
        assert_eq!(m[(2,0)], 3);

        assert_eq!(m[(0,1)], 4);
        assert_eq!(m[(1,1)], 5);
        assert_eq!(m[(2,1)], 6);

        assert_eq!(m[(0,2)], 7);
        assert_eq!(m[(1,2)], 8);
        assert_eq!(m[(2,2)], 9);

        assert_eq!(m[(0,3)], 10);
        assert_eq!(m[(1,3)], 11);
        assert_eq!(m[(2,3)], 12);

        assert_eq!(m[(0,4)], 13);
        assert_eq!(m[(1,4)], 14);
        assert_eq!(m[(2,4)], 15);
    }

    #[test]
    fn test_matrix_identity() {
        assert_eq!(
            Matrix3x3::<i8>::identity() * Matrix3x3::<i8>::identity(),
            Matrix3x3::<i8>::identity()
        );

        assert_eq!(
            Matrix3x3::<i8>::identity(),
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]].into()
        );

        assert_eq!(Matrix2x3::<i8>::identity(), [[1, 0], [0, 1], [0, 0]].into());
    }
}
