#pragma once

#include "conv2d_dyn.hpp"
#include "corr2d_dyn.hpp"

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 3D convolution output: odata(o) = sum(i, idata(i) @ kdata(i, o))
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void conv3d_output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        odata.setZero();

                        for (decltype(idata.dims()) i = 0, k = 0; i < idata.dims(); i ++)
                        {
                                for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++, k ++)
                                {
                                        auto omap = odata.matrix(o);
                                        auto imap = idata.matrix(i);
                                        auto kmap = kdata.matrix(k);

                                        math::conv2d_dyn(imap, kmap, omap);
                                }
                        }
                }

                ///
                /// \brief gradient wrt the input
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void conv3d_ginput(ttensori&& idata, const ttensork& kdata, const ttensoro& odata)
                {
                        idata.setZero();

                        for (decltype(idata.dims()) i = 0, k = 0; i < idata.dims(); i ++)
                        {
                                for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++, k ++)
                                {
                                        auto omap = odata.matrix(o);
                                        auto imap = idata.matrix(i);
                                        auto kmap = kdata.matrix(k);

                                        math::corr2d_dyn(omap, kmap, imap);
                                }
                        }
                }

                ///
                /// \brief gradient wrt the parameters
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void conv3d_gparam(const ttensori& idata, ttensork&& kdata, const ttensoro& odata)
                {
                        kdata.setZero();

                        for (decltype(idata.dims()) i = 0, k = 0; i < idata.dims(); i ++)
                        {
                                for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++, k ++)
                                {
                                        auto omap = odata.matrix(o);
                                        auto imap = idata.matrix(i);
                                        auto kmap = kdata.matrix(k);

                                        math::conv2d_dyn(imap, omap, kmap);
                                }
                        }
                }
        }
}


