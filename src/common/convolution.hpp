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
        }
}

#endif // NANOCV_CONVOLUTION_H

