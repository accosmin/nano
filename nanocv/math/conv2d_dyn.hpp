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
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv2d_dyn(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
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
                                case 1:         conv2d_doti<1>(idata, kdata, odata); break;
                                case 2:         conv2d_doti<2>(idata, kdata, odata); break;
                                case 3:         conv2d_doti<3>(idata, kdata, odata); break;
                                case 4:         conv2d_doti<4>(idata, kdata, odata); break;
                                case 5:         conv2d_doti<5>(idata, kdata, odata); break;
                                case 6:         conv2d_doti<6>(idata, kdata, odata); break;
                                case 7:         conv2d_doti<7>(idata, kdata, odata); break;
                                case 8:         conv2d_doti<8>(idata, kdata, odata); break;
                                case 9:         conv2d_doti<9>(idata, kdata, odata); break;
                                case 10:        conv2d_doti<10>(idata, kdata, odata); break;
                                case 11:        conv2d_doti<11>(idata, kdata, odata); break;
                                case 12:        conv2d_doti<12>(idata, kdata, odata); break;
                                case 13:        conv2d_doti<13>(idata, kdata, odata); break;
                                case 14:        conv2d_doti<14>(idata, kdata, odata); break;
                                case 15:        conv2d_doti<15>(idata, kdata, odata); break;
                                default:        conv2d_dot(idata, kdata, odata); break;
                                }
                        }
                        else
                        {
                                switch (ocols)
                                {
                                case 1:         conv2d_madi<1>(idata, kdata, odata); break;
                                case 2:         conv2d_madi<2>(idata, kdata, odata); break;
                                case 3:         conv2d_madi<3>(idata, kdata, odata); break;
                                case 4:         conv2d_madi<4>(idata, kdata, odata); break;
                                case 5:         conv2d_madi<5>(idata, kdata, odata); break;
                                case 6:         conv2d_madi<6>(idata, kdata, odata); break;
                                case 7:         conv2d_madi<7>(idata, kdata, odata); break;
                                case 8:         conv2d_madi<8>(idata, kdata, odata); break;
                                case 9:         conv2d_madi<9>(idata, kdata, odata); break;
                                case 10:        conv2d_madi<10>(idata, kdata, odata); break;
                                case 11:        conv2d_madi<11>(idata, kdata, odata); break;
                                case 12:        conv2d_madi<12>(idata, kdata, odata); break;
                                case 13:        conv2d_madi<13>(idata, kdata, odata); break;
                                case 14:        conv2d_madi<14>(idata, kdata, odata); break;
                                case 15:        conv2d_madi<15>(idata, kdata, odata); break;
                                default:        conv2d_mad(idata, kdata, odata); break;
                                }
                        }
                }
        }
}

