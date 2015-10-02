#pragma once

namespace tensor
{
        ///
        /// \brief 3D convolution output: odata(o) = sum(i, idata(i) @ kdata(i, o))
        ///
        template
        <
                typename toperator,     ///< 2D convolution operator
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        void conv3d_output(const toperator& conv2d_op,
                const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
        {
                odata.setZero();

                for (decltype(idata.dims()) i = 0, k = 0; i < idata.dims(); i ++)
                {
                        for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++, k ++)
                        {
                                auto omap = odata.matrix(o);
                                auto imap = idata.matrix(i);
                                auto kmap = kdata.matrix(k);

                                conv2d_op(imap, kmap, omap);
                        }
                }
        }

        ///
        /// \brief gradient wrt the input: odata(o) = sum(i, idata(i) @ kdata(i, o))
        ///
        template
        <
                typename toperator,     ///< 2D correlation operator
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        void conv3d_ginput(const toperator& corr2d_op,
                ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
        {
                idata.setZero();

                for (decltype(idata.dims()) i = 0, k = 0; i < idata.dims(); i ++)
                {
                        for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++, k ++)
                        {
                                auto omap = odata.matrix(o);
                                auto imap = idata.matrix(i);
                                auto kmap = kdata.matrix(k);

                                corr2d_op(omap, kmap, imap);
                        }
                }
        }

        ///
        /// \brief gradient wrt the parameters: odata(o) = sum(i, idata(i) @ kdata(i, o))
        ///
        template
        <
                typename toperator,     ///< 2D convolution operator
                typename ttensori,
                typename ttensork,
                typename ttensoro
        >
        void conv3d_gparam(const toperator& conv2d_op,
                const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
        {
                kdata.setZero();

                for (decltype(idata.dims()) i = 0, k = 0; i < idata.dims(); i ++)
                {
                        for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++, k ++)
                        {
                                auto omap = odata.matrix(o);
                                auto imap = idata.matrix(i);
                                auto kmap = kdata.matrix(k);

                                conv2d_op(imap, omap, kmap);
                        }
                }
        }
}


