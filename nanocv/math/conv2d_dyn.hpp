#pragma once

#include "conv2d_dot.hpp"
#include "conv2d_mad.hpp"

namespace ncv
{
        namespace math
        {
                ///
                /// \brief 2D convolution: odata += idata @ kdata (by decoding the kernel size at runtime)
                ///
                struct conv2d_dyn_t
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

                                const auto kcols = kdata.cols();
                                const auto ocols = odata.cols();

                                // decode at run-time the kernel size
                                if (kcols < ocols)
                                {
                                        switch (kcols)
                                        {
                                        case 3:         conv2d_doti_t<3>()(idata, kdata, odata); break;
                                        case 5:         conv2d_doti_t<5>()(idata, kdata, odata); break;
                                        case 7:         conv2d_doti_t<7>()(idata, kdata, odata); break;
                                        case 9:         conv2d_doti_t<9>()(idata, kdata, odata); break;
                                        case 11:        conv2d_doti_t<11>()(idata, kdata, odata); break;
                                        case 13:        conv2d_doti_t<13>()(idata, kdata, odata); break;
                                        case 15:        conv2d_doti_t<15>()(idata, kdata, odata); break;
                                        default:        conv2d_dot_t()(idata, kdata, odata); break;
                                        }
                                }
                                else
                                {
                                        switch (ocols)
                                        {
                                        case 3:         conv2d_madi_t<3>()(idata, kdata, odata); break;
                                        case 5:         conv2d_madi_t<5>()(idata, kdata, odata); break;
                                        case 7:         conv2d_madi_t<7>()(idata, kdata, odata); break;
                                        case 9:         conv2d_madi_t<9>()(idata, kdata, odata); break;
                                        case 11:        conv2d_madi_t<11>()(idata, kdata, odata); break;
                                        case 13:        conv2d_madi_t<13>()(idata, kdata, odata); break;
                                        case 15:        conv2d_madi_t<15>()(idata, kdata, odata); break;
                                        default:        conv2d_mad_t()(idata, kdata, odata); break;
                                        }
                                }
                        }
                };
        }
}

