#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <eigen3/Eigen/Core>

namespace ncv
{
        namespace math
        {
                namespace detail
                {
                       ///
                        /// 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
                        ///
                        template
                        <
                                int trows, int tcols,
                                typename tmatrix,
                                typename tscalar = typename tmatrix::Scalar
                        >
                        void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                        {
                                for (auto r = 0; r < odata.rows(); r ++)
                                {
                                        for (auto c = 0; c < odata.cols(); c ++)
                                        {
                                                odata(r, c) +=
                                                kdata.cwiseProduct(idata.block(r, c, trows, tcols)).sum();
                                        }
                                }
                        } 
                }
                
                ///
                /// 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        if (kdata.rows() == kdata.cols())
                        {
                                if (kdata.rows() == 4) { detail::conv<4, 4>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 5) { detail::conv<5, 5>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 6) { detail::conv<6, 6>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 7) { detail::conv<7, 7>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 8) { detail::conv<8, 8>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 9) { detail::conv<9, 9>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 10) { detail::conv<10, 10>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 11) { detail::conv<11, 11>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 12) { detail::conv<12, 12>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 13) { detail::conv<13, 13>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 14) { detail::conv<14, 14>(idata, kdata, odata); return; }
                                else if (kdata.rows() == 15) { detail::conv<15, 15>(idata, kdata, odata); return; }
                        }
                                
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        odata(r, c) +=
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                }
                        }
                        
                }

                ///
                /// 2D convolution: odata += weight * idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        odata(r, c) += weight *
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                }
                        }
                }                
        }
}

#endif // NANOCV_CONVOLUTION_H

