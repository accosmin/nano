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
                        typename ttensor
                >
                void output(const ttensor& idata, const ttensor& kdata, ttensor& odata)
                {
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (auto o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = odata.plane_matrix(o);

                                omap.setZero();
                                for (auto i = 0; i < idims; i ++, k ++)
                                {
                                        auto imap = idata.plane_matrix(i);
                                        auto kmap = kdata.plane_matrix(k);

                                        math::conv2d_dyn(imap, kmap, omap);
                                }
                        }
                }

                ///
                /// \brief gradient wrt the input
                ///
                template
                <
                        typename ttensor
                >
                void ginput(ttensor& gidata, const ttensor& kdata, const ttensor& odata)
                {
                        const auto idims = gidata.dims();
                        const auto odims = odata.dims();

                        gidata.setZero();
                        for (auto o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = odata.plane_matrix(o);

                                for (auto i = 0; i < idims; i ++, k ++)
                                {
                                        auto gimap = gidata.plane_matrix(i);
                                        auto kmap = kdata.plane_matrix(k);

                                        math::corr2d_dyn(omap, kmap, gimap);
                                }
                        }
                }

                ///
                /// \brief gradient wrt the kernel
                ///
                template
                <
                        typename ttensor
                >
                void gparam(const ttensor& idata, ttensor& gkdata, const ttensor& odata)
                {
                        const auto idims = idata.dims();
                        const auto odims = odata.dims();

                        for (auto o = 0, k = 0; o < odims; o ++)
                        {
                                auto omap = odata.plane_matrix(o);

                                for (auto i = 0; i < idims; i ++, k ++)
                                {
                                        auto imap = idata.plane_matrix(i);
                                        auto gkmap = gkdata.plane_matrix(k);

                                        gkmap.setZero();
                                        math::conv2d_dyn(imap, omap, gkmap);
                                }
                        }
                }
        }
}


