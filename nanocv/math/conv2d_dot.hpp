#pragma once

#include "dot.hpp"
#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a dot operator)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_dot(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
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
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                podata[c] += math::dot<tscalar>(pidata + c, pkdata, kcols);
                                        }
                                }
                        }
                }

                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a dot operator with a fixed kernel size)
                ///
                template
                <
                        int kcols,
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_doti(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto icols = idata.cols();

                        for (auto r = 0; r < orows; r ++)
                        {
                                tscalar* podata = odata.data() + r * ocols;

                                for (auto kr = 0; kr < krows; kr ++)
                                {
                                        const tscalar* pidata = idata.data() + (r + kr) * icols;
                                        const tscalar* pkdata = kdata.data() + kr * kcols;

                                        for (auto c = 0; c < ocols; c ++)
                                        {
                                                podata[c] += math::dot<kcols>(pidata + c, pkdata);
                                        }
                                }
                        }
                }
        }
}

