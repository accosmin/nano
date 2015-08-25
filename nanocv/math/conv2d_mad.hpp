#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a mad operator)
                ///
                struct conv2d_mad_t
                {
                        template
                        <
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi
                        >
                        void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata) const
                        {
                                assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                                assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                                const auto orows = odata.rows();
                                const auto ocols = odata.cols();
                                const auto krows = kdata.rows();
                                const auto kcols = kdata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        auto orow = odata.row(r);

                                        for (auto kr = 0; kr < krows; kr ++)
                                        {
                                                const auto irow = idata.row(r + kr);

                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        orow += irow.segment(kc, ocols) * kdata(kr, kc);
                                                }
                                        }
                                }
                        }
                };
        }
}

