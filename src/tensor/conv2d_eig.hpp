#pragma once

#include "conv2d_assert.hpp"

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
        ///
        struct conv2d_eig_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);

                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        odata(r, c) +=
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                }
                        }
                }
        };
}

