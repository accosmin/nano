#pragma once

#include "conv2d_toe.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief 3D convolution output: odata(o) = sum(i, idata(i) @ kdata(o, i))
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

                        for (decltype(idata.dims()) i = 0; i < idata.dims(); i ++)
                        {
                                auto imap = idata.matrix(i);

                                const auto toeplitz = tensor::make_toeplitz(imap, kdata.rows(), kdata.cols());

                                for (decltype(odata.dims()) o = 0; o < odata.dims(); o ++)
                                {
                                        auto omap = odata.matrix(o);
                                        auto kmap = kdata.matrix(o * idata.dims() + i);

                                        conv2d_toe_buffered(imap, kmap, toeplitz, omap);
                                }
                        }
                }

//                ///
//                /// \brief gradient wrt the input
//                ///
//                template
//                <
//                        typename ttensori,
//                        typename ttensork,
//                        typename ttensoro
//                >
//                void conv3d_ginput(ttensori&& gidata, const ttensork& kdata, const ttensoro& odata)
//                {
//                        gidata.setZero();

//                        for (decltype(odata.dims()) o = 0, k = 0; o < odata.dims(); o ++)
//                        {
//                                auto omap = odata.matrix(o);

//                                for (decltype(gidata.dims()) i = 0; i < gidata.dims(); i ++, k ++)
//                                {
//                                        auto gimap = gidata.matrix(i);
//                                        auto kmap = kdata.matrix(k);

//                                        math::corr2d_dyn(omap, kmap, gimap);
//                                }
//                        }
//                }

//                ///
//                /// \brief gradient wrt the parameters
//                ///
//                template
//                <
//                        typename ttensori,
//                        typename ttensork,
//                        typename ttensoro
//                >
//                void conv3d_gparam(const ttensori& idata, ttensork&& gkdata, const ttensoro& odata)
//                {
//                        for (decltype(odata.dims()) o = 0, k = 0; o < odata.dims(); o ++)
//                        {
//                                auto omap = odata.matrix(o);

//                                for (decltype(idata.dims()) i = 0; i < idata.dims(); i ++, k ++)
//                                {
//                                        auto imap = idata.matrix(i);
//                                        auto gkmap = gkdata.matrix(k);

//                                        gkmap.setZero();
//                                        math::conv2d_dyn(imap, omap, gkmap);
//                                }
//                        }
//                }
        }
}


