#pragma once

namespace ncv
{        
        ///
        /// \brief general mad-product
        ///
        template
        <
                typename tscalar
        >
        void mad(const tscalar* idata, tscalar weight, int n, tscalar* odata)
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
                typename tscalar,
                int tsize
        >
        void mad(const tscalar* idata, tscalar weight, int, tscalar* odata)
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
        void mad_unroll(const tscalar* idata, tscalar weight, int n, tscalar* odata)
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
