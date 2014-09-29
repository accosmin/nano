#pragma once

namespace ncv
{        
        ///
        /// \brief general dot-product
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        tscalar dot(const tscalar* vec1, const tscalar* vec2, tsize size)
        {
                tscalar sum = 0;
                for (auto i = 0; i < size; i ++)
                {
                        sum += vec1[i] * vec2[i];
                }

                return sum;
        }

        ///
        /// \brief general dot-product (unrolled by 2)
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        tscalar dot_unroll2(const tscalar* vec1, const tscalar* vec2, tsize size)
        {
                const tsize size2 = size & tsize(~1);

                tscalar sum0 = 0, sum1 = 0;
                for (auto i = 0; i < size2; i += 2)
                {
                        sum0 += vec1[i + 0] * vec2[i + 0];
                        sum1 += vec1[i + 1] * vec2[i + 1];
                }
                for (auto i = size2; i < size; i ++)
                {
                        sum0 += vec1[i] * vec2[i];
                }

                return (sum0 + sum1);
        }

        ///
        /// \brief general dot-product (unrolled by 4)
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        tscalar dot_unroll4(const tscalar* vec1, const tscalar* vec2, tsize size)
        {
                const tsize size4 = size & tsize(~3);

                tscalar sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
                for (auto i = 0; i < size4; i += 4)
                {
                        sum0 += vec1[i + 0] * vec2[i + 0];
                        sum1 += vec1[i + 1] * vec2[i + 1];
                        sum2 += vec1[i + 2] * vec2[i + 2];
                        sum3 += vec1[i + 3] * vec2[i + 3];
                }
                for (auto i = size4; i < size; i ++)
                {
                        sum0 += vec1[i] * vec2[i];
                }

                return (sum0 + sum1) + (sum2 + sum3);
        }

        ///
        /// \brief general dot-product (unrolled by 8)
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        tscalar dot_unroll8(const tscalar* vec1, const tscalar* vec2, tsize size)
        {
                const tsize size8 = size & tsize(~7);
                const tsize size4 = size & tsize(~3);

                tscalar sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0, sum4 = 0, sum5 = 0, sum6 = 0, sum7 = 0;
                for (auto i = 0; i < size8; i += 8)
                {
                        sum0 += vec1[i + 0] * vec2[i + 0];
                        sum1 += vec1[i + 1] * vec2[i + 1];
                        sum2 += vec1[i + 2] * vec2[i + 2];
                        sum3 += vec1[i + 3] * vec2[i + 3];
                        sum4 += vec1[i + 4] * vec2[i + 4];
                        sum5 += vec1[i + 5] * vec2[i + 5];
                        sum6 += vec1[i + 6] * vec2[i + 6];
                        sum7 += vec1[i + 7] * vec2[i + 7];
                }
                for (auto i = size8; i < size4; i += 4)
                {
                        sum0 += vec1[i + 0] * vec2[i + 0];
                        sum1 += vec1[i + 1] * vec2[i + 1];
                        sum2 += vec1[i + 2] * vec2[i + 2];
                        sum3 += vec1[i + 3] * vec2[i + 3];
                }
                for (auto i = size4; i < size; i ++)
                {
                        sum0 += vec1[i] * vec2[i];
                }

                return (sum0 + sum1) + (sum2 + sum3) + (sum4 + sum5) + (sum6 + sum7);
        }

        ///
        /// \brief fixed-size dot-product
        ///
        template
        <
                typename tscalar,
                int tsize
        >
        tscalar dot(const tscalar* vec1, const tscalar* vec2, int)
        {
                tscalar sum = 0;
                for (auto i = 0; i < tsize; i ++)
                {
                        sum += vec1[i] * vec2[i];
                }

                return sum;
        }
}
