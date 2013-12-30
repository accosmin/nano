#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include "dot.hpp"
#include <algorithm>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // 2D convolution utilities.
        /////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                namespace impl
                {
                        // odata = idata @ kdata (using a column-based dot operator)
                        template
                        <
                                bool tcumulate,
                                typename tmatrix,
                                typename tdotop,         // column-based dot operator
                                typename tscalar = typename tmatrix::Scalar,
                                typename tindex = typename tmatrix::Index
                        >
                        void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata, tdotop dop)
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        tscalar* podata = &odata(r, 0);

                                        if (!tcumulate)
                                        {
                                                std::fill(podata, podata + odata.cols(), 0);
                                        }

                                        for (tindex kr = 0; kr < kdata.rows(); kr ++)
                                        {
                                                const tscalar *pidata = &idata(r + kr, 0), *pkdata = &kdata(kr, 0);

                                                for (tindex c = 0; c < odata.cols(); c ++)
                                                {
                                                        podata[c] += dop(pidata + c, pkdata, kdata.cols());
                                                }
                                        }
                                }
                        }

                        // odata = weight * idata @ kdata (using a column-based dot operator)
                        template
                        <
                                bool tcumulate,
                                typename tmatrix,
                                typename tdotop,         // column-based dot operator
                                typename tscalar = typename tmatrix::Scalar,
                                typename tindex = typename tmatrix::Index
                        >
                        void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata, tdotop dop)
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        tscalar* podata = &odata(r, 0);

                                        if (!tcumulate)
                                        {
                                                std::fill(podata, podata + odata.cols(), 0);
                                        }

                                        for (tindex kr = 0; kr < kdata.rows(); kr ++)
                                        {
                                                const tscalar *pidata = &idata(r + kr, 0), *pkdata = &kdata(kr, 0);

                                                for (tindex c = 0; c < odata.cols(); c ++)
                                                {
                                                        podata[c] += weight * dop(pidata + c, pkdata, kdata.cols());
                                                }
                                        }
                                }
                        }
                }

                // 2D convolution: odata = (weight *) idata @ kdata
                //      loop unrolling using 4 operations
                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_mod4(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot_mod4<tscalar>);
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_mod4(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot_mod4<tscalar>);
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_mod4x(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot_mod4x<tscalar>);
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_mod4x(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot_mod4x<tscalar>);
                }

                // 2D convolution: odata = (weight *) idata @ kdata
                //      no loop unrolling
                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot<tscalar>);
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot<tscalar>);
                }
                
                // 2D convolution: odata = (weight *) idata @ kdata
                //      use Eigen for dot product
                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_eigen(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot_eigen<tscalar>);
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_eigen(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot_eigen<tscalar>);
                }

                // 2D convolution: odata = (weight *) idata @ kdata
                //      using Eigen blocks
                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar,
                        typename tindex = typename tmatrix::Index
                >
                void conv_eigen_block(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        if (tcumulate)
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        for (tindex c = 0; c < odata.cols(); c ++)
                                        {
                                                odata(r, c) +=
                                                kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                        }
                               }
                        }

                        else
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        for (tindex c = 0; c < odata.cols(); c ++)
                                        {
                                                odata(r, c) =
                                                kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                        }
                               }
                        }
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar,
                        typename tindex = typename tmatrix::Index
                >
                void wconv_eigen_block(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        if (tcumulate)
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        for (tindex c = 0; c < odata.cols(); c ++)
                                        {
                                                odata(r, c) += weight *
                                                kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();
                                        }
                                }
                        }

                        else
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        for (tindex c = 0; c < odata.cols(); c ++)
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

