#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <eigen3/Eigen/Core>
#include <algorithm>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 2D convolution.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // odata = idata @ kdata (using a column-based dot operator)
                template
                <
                        bool tcumulate,
                        typename tdot,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_dot(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata, tdot dotop)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                tscalar* podata = &odata(r, 0);

                                if (!tcumulate)
                                {
                                        std::fill(podata, podata + odata.cols(), 0);
                                }

                                for (auto kr = 0; kr < kdata.rows(); kr ++)
                                {
                                        const tscalar *pidata = &idata(r + kr, 0), *pkdata = &kdata(kr, 0);

                                        for (auto c = 0; c < odata.cols(); c ++)
                                        {
                                                podata[c] += dotop(pidata + c, pkdata, kdata.cols());
                                        }
                                }
                        }
                }

                // odata = weight * idata @ kdata (using a column-based dot operator)
                template
                <
                        bool tcumulate,
                        typename tdot,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_dot(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata, tdot dotop)
                {
                        for (auto r = 0; r < odata.rows(); r ++)
                        {
                                tscalar* podata = &odata(r, 0);

                                if (!tcumulate)
                                {
                                        std::fill(podata, podata + odata.cols(), 0);
                                }

                                for (auto kr = 0; kr < kdata.rows(); kr ++)
                                {
                                        const tscalar *pidata = &idata(r + kr, 0), *pkdata = &kdata(kr, 0);

                                        for (auto c = 0; c < odata.cols(); c ++)
                                        {
                                                podata[c] += weight * dotop(pidata + c, pkdata, kdata.cols());
                                        }
                                }
                        }
                }
                
                // odata = idata @ kdata (using Eigen 2D blocks)
                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_eib(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        if (tcumulate)
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

                        else
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
                }

                // odata = weight * idata @ kdata (using Eigen 2D blocks)
                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_eib(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        if (tcumulate)
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

                        else
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
}

#endif // NANOCV_CONVOLUTION_H

