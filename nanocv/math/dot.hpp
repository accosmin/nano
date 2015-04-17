#pragma once

#include "nanocv/arch.h"

namespace ncv
{        
        namespace math
        {
                ///
                /// \brief general dot-product
                ///
                template
                <
                        typename tscalar
                >
                tscalar dot(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        tscalar sum = 0;
                        for (int i = 0; i < n; i ++)
                        {
                                sum += a[i] * b[i];
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
                tscalar dot(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int)
                {
                        tscalar sum = 0;
                        for (int i = 0; i < tsize; i ++)
                        {
                                sum += a[i] * b[i];
                        }

                        return sum;
                }

                ///
                /// \brief unrolled dot-product
                ///
                template
                <
                        typename tscalar,
                        int tunrollsize
                >
                tscalar dot_unroll(const tscalar* NANOCV_RESTRICT a, const tscalar* NANOCV_RESTRICT b, int n)
                {
                        int i = 0;

                        tscalar sum = 0;
                        for ( ; i + tunrollsize < n; i += tunrollsize)
                        {
                                for (int t = 0; t < tunrollsize; t ++)
                                {
                                        sum += a[i + t] * b[i + t];
                                }
                        }
                        for ( ; i < n; i ++)
                        {
                                sum += a[i] * b[i];
                        }

                        return sum;
                }
        }
}
