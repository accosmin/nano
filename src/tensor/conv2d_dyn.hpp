#pragma once

#include "conv2d_dot.hpp"
#include "conv2d_mad.hpp"

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (decode at runtime the best operator to use)
        ///
        struct conv2d_dyn_t
        {
                template
                <
                        typename tmatrixi,
                        typename tmatrixk,
                        typename tmatrixo
                >
                void operator()(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo&& odata) const
                {
                        conv2d_assert(idata, kdata, odata);

                        if (kdata.rows() == 1 && kdata.cols() == 1)
                        {
                                odata.noalias() += idata * kdata(0, 0);
                        }
                        else if (odata.rows() == 1 && odata.cols() == 1)
                        {
                                odata(0, 0) += (idata.array() * kdata.array()).sum();
                        }
                        else
                        {
                                conv2d_dot_dyn_t()(idata, kdata, odata);
                        }
                }
        };
}

