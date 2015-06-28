#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D correlation: idata += odata @ kdata (using plain array indexing)
                ///
                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr2d_cpp(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                const tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        pidata[c + kc] += pkdata[kc] * podata[c];
                                                }
                                        }
                                }
                        }
                }
        }
}

