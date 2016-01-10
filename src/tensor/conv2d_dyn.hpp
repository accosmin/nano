#pragma once

#include "conv2d_dot.hpp"
#include "conv2d_mad.hpp"

namespace tensor
{
        ///
        /// \brief 2D convolution: odata += idata @ kdata (by decoding the kernel size at runtime)
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
                        assert(idata.rows() + 1 == kdata.rows() + odata.rows());
                        assert(idata.cols() + 1 == kdata.cols() + odata.cols());

                        const auto kcols = kdata.cols();
                        const auto ocols = odata.cols();

                        // decode at run-time the kernel size
                        if (kcols < 15)
                        {
                                switch (kcols)
                                {
                                case 1:         conv2d_doti_t<1>()(idata, kdata, odata); break;
                                case 2:         conv2d_doti_t<2>()(idata, kdata, odata); break;
                                case 3:         conv2d_doti_t<3>()(idata, kdata, odata); break;
                                case 4:         conv2d_doti_t<4>()(idata, kdata, odata); break;
                                case 5:         conv2d_doti_t<5>()(idata, kdata, odata); break;
                                case 6:         conv2d_doti_t<6>()(idata, kdata, odata); break;
                                case 7:         conv2d_doti_t<7>()(idata, kdata, odata); break;
                                case 8:         conv2d_doti_t<8>()(idata, kdata, odata); break;
                                case 9:         conv2d_doti_t<9>()(idata, kdata, odata); break;
                                case 10:        conv2d_doti_t<10>()(idata, kdata, odata); break;
                                case 11:        conv2d_doti_t<11>()(idata, kdata, odata); break;
                                case 12:        conv2d_doti_t<12>()(idata, kdata, odata); break;
                                case 13:        conv2d_doti_t<13>()(idata, kdata, odata); break;
                                case 14:        conv2d_doti_t<14>()(idata, kdata, odata); break;
                                case 15:        conv2d_doti_t<15>()(idata, kdata, odata); break;
                                default:        conv2d_dot_t()(idata, kdata, odata); break;
                                }
                        }
                        else if (ocols < 15)
                        {
                                switch (ocols)
                                {
                                case 1:         conv2d_madi_t<1>()(idata, kdata, odata); break;
                                case 2:         conv2d_madi_t<2>()(idata, kdata, odata); break;
                                case 3:         conv2d_madi_t<3>()(idata, kdata, odata); break;
                                case 4:         conv2d_madi_t<4>()(idata, kdata, odata); break;
                                case 5:         conv2d_madi_t<5>()(idata, kdata, odata); break;
                                case 6:         conv2d_madi_t<6>()(idata, kdata, odata); break;
                                case 7:         conv2d_madi_t<7>()(idata, kdata, odata); break;
                                case 8:         conv2d_madi_t<8>()(idata, kdata, odata); break;
                                case 9:         conv2d_madi_t<9>()(idata, kdata, odata); break;
                                case 10:        conv2d_madi_t<10>()(idata, kdata, odata); break;
                                case 11:        conv2d_madi_t<11>()(idata, kdata, odata); break;
                                case 12:        conv2d_madi_t<12>()(idata, kdata, odata); break;
                                case 13:        conv2d_madi_t<13>()(idata, kdata, odata); break;
                                case 14:        conv2d_madi_t<14>()(idata, kdata, odata); break;
                                case 15:        conv2d_madi_t<15>()(idata, kdata, odata); break;
                                default:        conv2d_mad_t()(idata, kdata, odata); break;
                                }
                        }
                        else 
                        {
                                conv2d_dot_t()(idata, kdata, odata);
                        }
                }
        };
}

