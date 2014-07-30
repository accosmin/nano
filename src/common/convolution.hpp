#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <eigen3/Eigen/Core>

namespace ncv
{
        namespace math
        {
                ///
                /// 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_sum(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
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
                /// 2D convolution: odata = idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        odata(r, c) =
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                }
                        }                        
                }
                
                ///
                /// 2D convolution: odata = idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void conv_fast(const tmatrixi& idata, const tmatrixk& kdata, tmatrixo& odata)
                {
                        const auto orows = odata.rows();
                        const auto ocols = odata.cols();
                        const auto krows = kdata.rows();
                        const auto kcols = kdata.cols();
                        
                        for (auto r = 0; r < orows; r ++)
                        {
                                for (auto c = 0; c < ocols; c ++)
                                {
                                        tscalar sum = 0;
                                        for (auto kr = 0; kr < krows; kr ++)
                                        {
                                                for (auto kc = 0; kc < kcols; kc ++)
                                                {
                                                        sum += idata(r + kr, c + kc) * kdata(kr, kc);
                                                }
                                        }
                                        
                                        odata(r, c) = sum;
                                }
                        }                     
                }

                ///
                /// 2D convolution: odata += weight * idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void wconv_sum(const tmatrixi& idata, const tmatrixk& kdata, tscalar weight, tmatrixo& odata)
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
                
                ///
                /// 2D convolution: odata = weight * idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void wconv(const tmatrixi& idata, const tmatrixk& kdata, tscalar weight, tmatrixo& odata)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        odata(r, c) = weight *
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                }
                        }
                } 
                
                ///
                /// outer 2D convolution: odata += idata @ kdata (using Eigen 2D blocks)
                ///
                template
                <
                        typename tmatrixi,
                        typename tmatrixk = tmatrixi,
                        typename tmatrixo = tmatrixi,
                        typename tscalar = typename tmatrixi::Scalar
                >
                void outer_conv_sum(tmatrixi& idata, const tmatrixk& kdata, const tmatrixo& odata)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                for (auto c = 0; c < odata.cols(); c ++)
                                {
                                        idata.block(r, c, kdata.rows(), kdata.cols()) += kdata * odata(r, c);
                                }
                        }
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

