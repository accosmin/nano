#ifndef NANOCV_DOT_H
#define NANOCV_DOT_H

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

                tscalar sum = 0;
                for (auto i = 0; i < size4; i += 4)
                {
                        sum += vec1[i + 0] * vec2[i + 0] +
                               vec1[i + 1] * vec2[i + 1] +
                               vec1[i + 2] * vec2[i + 2] +
                               vec1[i + 3] * vec2[i + 3];
                }
                for (auto i = size4; i < size; i ++)
                {
                        sum += vec1[i] * vec2[i];
                }

                return sum;
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
                const tsize size4 = size & tsize(~4);

                tscalar sum = 0;
                for (auto i = 0; i < size8; i += 8)
                {
                        sum += vec1[i + 0] * vec2[i + 0] +
                               vec1[i + 1] * vec2[i + 1] +
                               vec1[i + 2] * vec2[i + 2] +
                               vec1[i + 3] * vec2[i + 3] +
                               vec1[i + 4] * vec2[i + 4] +
                               vec1[i + 5] * vec2[i + 5] +
                               vec1[i + 6] * vec2[i + 6] +
                               vec1[i + 7] * vec2[i + 7];
                }
                for (auto i = size8; i < size4; i += 4)
                {
                        sum += vec1[i + 0] * vec2[i + 0] +
                               vec1[i + 1] * vec2[i + 1] +
                               vec1[i + 2] * vec2[i + 2] +
                               vec1[i + 3] * vec2[i + 3];
                }
                for (auto i = size4; i < size; i ++)
                {
                        sum += vec1[i] * vec2[i];
                }

                return sum;
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

#endif // NANOCV_DOT_H

