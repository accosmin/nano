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
                                const auto icols = idata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        auto* podata = odata.data() + r * ocols;

                                        for (auto kr = 0; kr < krows; kr ++)
                                        {
                                                const auto* pidata = idata.data() + (r + kr) * icols;
                                                const auto* pkdata = kdata.data() + kr * kcols;

                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        math::mad(pidata + kc, pkdata[kc], ocols, podata);
                                                }
                                        }
                                }
                        }
                };

                ///
                /// \brief 2D convolution: odata += idata @ kdata (using a mad operator for a fixed output size)
                ///
                template
                <
                        int ocols
                >
                struct conv2d_madi_t
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
                                const auto krows = kdata.rows();
                                const auto kcols = kdata.cols();
                                const auto icols = idata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        auto* podata = odata.data() + r * ocols;

                                        for (auto kr = 0; kr < krows; kr ++)
                                        {
                                                const auto* pidata = idata.data() + (r + kr) * icols;
                                                const auto* pkdata = kdata.data() + kr * kcols;

                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        math::mad<ocols>(pidata + kc, pkdata[kc], podata);
                                                }
                                        }
                                }
                        }
                };
        }
}

