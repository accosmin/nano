#pragma once

#include "corr2d_mdk.hpp"
#include "corr2d_mdo.hpp"

namespace tensor
{
        ///
        /// \brief 2D correlation: idata += odata @ kdata (decode at runtime the best operator to use)
        ///
        struct corr2d_dyn_t
        {
                template
                <
                        typename tmatrixo,
                        typename tmatrixk,
                        typename tmatrixi
                >
                void operator()(const tmatrixo& odata, const tmatrixk& kdata, tmatrixi&& idata) const
                {
                        conv2d_assert(idata, kdata, odata);

                        if (kdata.rows() == 1 && kdata.cols() == 1)
                        {
                                idata.noalias() += kdata(0, 0) * odata;
                        }
                        else if (odata.rows() == 1 && odata.cols() == 1)
                        {
                                idata.noalias() += odata(0, 0) * kdata;
                        }
                        else
                        {
                                corr2d_mdk_dyn_t()(odata, kdata, idata);
                        }
                }
        };
}

