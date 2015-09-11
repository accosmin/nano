#pragma once

#include <cassert>

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D correlation: idata += odata @ kdata (using a fixed-size mad product by kdata columns)
                ///
                template
                <
                        int kcols
                >
                struct corr2d_mdki_t
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
                                assert(kdata.cols() == kcols);

                                const auto orows = odata.rows();
                                const auto ocols = odata.cols();
                                const auto krows = kdata.rows();

                                for (auto r = 0; r < orows; r ++)
                                {
                                        for (auto kr = 0; kr < krows; kr ++)
                                        {
                                                const auto krow = kdata.row(kr);

                                                for (auto c = 0; c < ocols; c ++)
                                                {
                                                        idata.row(r + kr).template segment<kcols>(c) +=
                                                        krow * odata(r, c);
                                                }
                                        }
                                }
                        }
                };
        }
}

