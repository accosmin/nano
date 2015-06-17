#pragma once

#include "nanocv/arch.h"

namespace ncv
{        
        namespace math
        {
                ///
                /// \brief general mad-product
                ///
                template
                <
                        typename tscalar
                >
                void mad(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        for (int i = 0; i < n; i ++)
                        {
                                odata[i] += idata[i] * weight;
                        }
                }

                ///
                /// \brief fixed-size mad-product
                ///
                template
                <
                        int tsize,
                        typename tscalar
                >
                void mad(const tscalar* NANOCV_RESTRICT idata, tscalar weight, tscalar* NANOCV_RESTRICT odata)
                {
                        for (int i = 0; i < tsize; i ++)
                        {
                                odata[i] += idata[i] * weight;
                        }
                }

                ///
                /// \brief unrolled mad-product
                ///
                template
                <
                        typename tscalar,
                        int tunrollsize
                >
                void mad_unroll(const tscalar* NANOCV_RESTRICT idata, tscalar weight, int n, tscalar* NANOCV_RESTRICT odata)
                {
                        int i = 0;
                        for ( ; i + tunrollsize < n; i += tunrollsize)
                        {
                                for (int t = 0; t < tunrollsize; t ++)
                                {
                                        odata[i + t] += idata[i + t] * weight;
                                }
                        }
                        for ( ; i < n; i ++)
                        {
                                odata[i] += idata[i] * weight;
                        }
                }
        }
}
