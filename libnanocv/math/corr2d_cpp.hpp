#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D correlation: idata += odata @ kdata (using plain array indexing)
                ///
                struct corr2d_cpp_t
                {
                        template
                        <
                                typename tmatrixo,
                                typename tmatrixk = tmatrixo,
                                typename tmatrixi = tmatrixo
                        >
                        void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata) const
                        {
                                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

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
}

