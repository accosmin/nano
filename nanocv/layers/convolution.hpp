#pragma once

#include "nanocv/math/conv2d.hpp"
#include "nanocv/math/corr2d.hpp"
#include "nanocv/tensor/matrix.hpp"

namespace ncv
{
        namespace convolution
        {
                ///
                /// \brief convolution output
                ///
                template
                <
                        typename ttensori,
                        typename ttensork,
                        typename ttensoro
                >
                void output(const ttensori& idata, const ttensork& kdata, ttensoro&& odata)
                {
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (decltype(odata.dims()) o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = odata.matrix(o);

                                omap.setZero();
                                for (decltype(idata.dims()) i = 0; i < idims; i ++, k ++)
                                {
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
                void ginput(ttensori&& gidata, const ttensork& kdata, const ttensoro& odata)
                {
                        const auto idims = gidata.dims();
                        const auto odims = odata.dims();

                        gidata.setZero();
                        for (decltype(odata.dims()) o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = odata.matrix(o);

                                for (decltype(gidata.dims()) i = 0; i < idims; i ++, k ++)
                                {
                                        auto gimap = gidata.matrix(i);
                                        auto kmap = kdata.matrix(k);

                                        math::corr2d_dyn(omap, kmap, gimap);
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
                void gparam(const ttensori& idata, ttensork&& gkdata, const ttensoro& odata)
                {
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (decltype(odata.dims()) o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = odata.matrix(o);

                                for (decltype(idata.dims()) i = 0; i < idims; i ++, k ++)
                                {
                                        auto imap = idata.matrix(i);
                                        auto gkmap = gkdata.matrix(k);

                                        gkmap.setZero();
                                        math::conv2d_dyn(imap, omap, gkmap);
                                }
                        }
                }
        }
}


