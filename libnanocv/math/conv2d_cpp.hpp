#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using plain array indexing)
                ///
                struct conv2d_cpp_t
                {
                        template
                        <
                                typename tmatrixi,
                                typename tmatrixk = tmatrixi,
                                typename tmatrixo = tmatrixi,
                                typename tscalar = typename tmatrixo::Scalar
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
}

