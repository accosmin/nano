#pragma once

#include "conv2d_assert.hpp"

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (using plain array indexing)
        ///
        struct conv2d_cpp_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                tscalar sum = 0;
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        sum += idata(r + kr, c + kc) * kdata(kr, kc);
                                                }
                                                odata(r, c) += sum;
                                        }
                                }
                        }
                }
        };
}

