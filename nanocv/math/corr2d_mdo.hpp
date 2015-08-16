#pragma once

#include "mad.hpp"
#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D correlation: idata += odata @ kdata (using a mad product by odata columns)
                ///
                struct corr2d_mdo_t
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
                                const auto icols = idata.cols();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        const auto* podata = odata.data() + r * ocols;

                                        for (auto kr = 0; kr < krows; kr ++)
                                        {
                                                auto* pidata = idata.data() + (r + kr) * icols;
                                                const auto* pkdata = kdata.data() + kr * kcols;

                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        math::mad(podata, pkdata[kc], ocols, pidata + kc);
                                                }
                                        }
                                }
                        }
                };
        }
}

