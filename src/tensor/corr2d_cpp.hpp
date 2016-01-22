#pragma once

#include "conv2d_assert.hpp"

namespace tensor
{
        ///
        /// \brief 2D correlation: idata += odata @ kdata (using plain array indexing)
        ///
        struct corr2d_cpp_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk,
                        typename tmatrixi
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi&& idata) const
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
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        idata(r + kr, c + kc) += kdata(kr, kc) * odata(r, c);
                                                }
                                        }
                                }
                        }
                }
        };
}

