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

