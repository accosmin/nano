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
        /// \brief general mad-product (unroll by 4)
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        void mad_unroll4(const tscalar* idata, tscalar weight, tsize size, tscalar* odata)
        {
                const tsize size4 = size - (size & 3);

                for (auto i = 0; i < size4; i += 4)
                {
                        odata[i + 0] += idata[i + 0] * weight;
                        odata[i + 1] += idata[i + 1] * weight;
                        odata[i + 2] += idata[i + 2] * weight;
                        odata[i + 3] += idata[i + 3] * weight;
                }
                for (auto i = size4; i < size; i ++)
                {
                        odata[i] += idata[i] * weight;
                }
        }

        ///
        /// \brief general mad-product (unroll by 8)
        ///
        template
        <
                typename tscalar,
                typename tsize
        >
        void mad_unroll8(const tscalar* idata, tscalar weight, tsize size, tscalar* odata)
        {
                const tsize size8 = size - (size & 7);
                const tsize size4 = size - (size & 3);

                for (auto i = 0; i < size8; i += 8)
                {
                        odata[i + 0] += idata[i + 0] * weight;
                        odata[i + 1] += idata[i + 1] * weight;
                        odata[i + 2] += idata[i + 2] * weight;
                        odata[i + 3] += idata[i + 3] * weight;
                        odata[i + 4] += idata[i + 4] * weight;
                        odata[i + 5] += idata[i + 5] * weight;
                        odata[i + 6] += idata[i + 6] * weight;
                        odata[i + 7] += idata[i + 7] * weight;
                }
                for (auto i = size8; i < size4; i += 4)
                {
                        odata[i + 0] += idata[i + 0] * weight;
                        odata[i + 1] += idata[i + 1] * weight;
                        odata[i + 2] += idata[i + 2] * weight;
                        odata[i + 3] += idata[i + 3] * weight;
                }
                for (auto i = size4; i < size; i ++)
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
                for (auto i = 0; i < tsize; i ++)
                {
                        odata[i] += idata[i] * weight;
                }
        }
}

#endif // NANOCV_MAD_H

