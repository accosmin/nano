#ifndef NANOCV_MAD_H
#define NANOCV_MAD_H

namespace ncv
{        
        ///
        /// \brief general mad-product
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        void mad(const tscalar* idata, tscalar weight, tsize size, tscalar* odata)
        {
                for (auto i = 0; i < size; i ++)
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
        void mad(const tscalar* idata, tscalar weight, tscalar* odata)
        {
                for (auto i = 0; i < tsize; i ++)
                {
                        odata[i] += idata[i] * weight;
                }
        }
}

#endif // NANOCV_MAD_H

