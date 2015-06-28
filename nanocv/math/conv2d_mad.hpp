#pragma once

#include "mad.hpp"
#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a mad operator)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_mad(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
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

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                math::mad<tscalar>(pidata + kc, pkdata[kc], ocols, podata);
                                        }
                                }
                        }
                }

                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a mad operator for a fixed output size)
                ///
                template
                <
                        int ocols,
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_madi(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto orows = odata.rows();
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

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                math::mad<ocols>(pidata + kc, pkdata[kc], podata);
                                        }
                                }
                        }
                }
        }
}

