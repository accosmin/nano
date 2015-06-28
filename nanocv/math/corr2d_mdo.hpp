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
                template
                <
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr2d_mdo(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
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

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                math::mad<tscalar>(podata, pkdata[kc], ocols, pidata + kc);
                                        }
                                }
                        }
                }

                ///
                /// \brief 2D correlation: idata += odata @ kdata (using a fixed-size mad product by odata columns)
                ///
                template
                <
                        int ocols,
                        typename tmatrixo,
                        typename tmatrixk = tmatrixo,
                        typename tmatrixi = tmatrixo,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void corr2d_mdoi(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi& idata)
                {
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto orows = odata.rows();
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

                                        for (auto kc = 0; kc < kcols; kc ++)
                                        {
                                                math::mad<ocols>(podata, pkdata[kc], pidata + kc);
                                        }
                                }
                        }
                }
        }
}

