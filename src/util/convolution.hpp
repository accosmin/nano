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
                // implementation detail
                namespace impl
                {
                        template
                        <
                                bool tcumulate,
                                typename tmatrix,
                                typename tdotop,         // column-based dot operator
                                typename tscalar = typename tmatrix::Scalar,
                                typename tindex = typename tmatrix::Index
                        >
                        void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata,
                                  const tdotop& dop, tindex krows, tindex kcols)
                        {                                
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        tscalar* podata = &odata(r, 0);

                                        if (!tcumulate)
                                        {
                                                std::fill(podata, podata + odata.cols(), 0);
                                        }

                                        for (tindex kr = 0; kr < krows; kr ++)
                                        {
                                                const tscalar *pidata = &idata(r + kr, 0), *pkdata = &kdata(kr, 0);

                                                for (tindex c = 0; c < odata.cols(); c ++)
                                                {
                                                        podata[c] += dop(pidata + c, pkdata, kcols);
                                                }
                                        }
                                }
                        }

                        template
                        <
                                bool tcumulate,
                                typename tmatrix,
                                typename tdotop,         // column-based dot operator
                                typename tscalar = typename tmatrix::Scalar,
                                typename tindex = typename tmatrix::Index
                        >
                        void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata,
                                   const tdotop& dop, tindex krows, tindex kcols)
                        {
                                for (tindex r = 0; r < odata.rows(); r ++)
                                {
                                        tscalar* podata = &odata(r, 0);

                                        if (!tcumulate)
                                        {
                                                std::fill(podata, podata + odata.cols(), 0);
                                        }

                                        for (tindex kr = 0; kr < krows; kr ++)
                                        {
                                                const tscalar *pidata = &idata(r + kr, 0), *pkdata = &kdata(kr, 0);

                                                for (tindex c = 0; c < odata.cols(); c ++)
                                                {
                                                        podata[c] += weight * dop(pidata + c, pkdata, kcols);
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
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot_mod4<tscalar>, kdata.rows(), kdata.cols());
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_mod4(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot_mod4<tscalar>, kdata.rows(), kdata.cols());
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
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot<tscalar>, kdata.rows(), kdata.cols());
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot<tscalar>, kdata.rows(), kdata.cols());
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
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot_eigen<tscalar>, kdata.rows(), kdata.cols());
                }

                template
                <
                        bool tcumulate,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_eigen(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot_eigen<tscalar>, kdata.rows(), kdata.cols());
                }

                // 2D convolution: odata = (weight *) idata @ kdata
                //      for fixed size convolution (number of rows & columns)
                template
                <
                        bool tcumulate,
                        int tkrows,
                        int tkcols,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv<tcumulate>(idata, kdata, odata, math::dot<tkcols, tscalar>, tkrows, tkcols);
                }

                template
                <
                        bool tcumulate,
                        int tkrows,
                        int tkcols,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv<tcumulate>(idata, kdata, weight, odata, math::dot<tkcols, tscalar>, tkrows, tkcols);
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
                        for (tindex r = 0; r < odata.rows(); r ++)
                        {
                                for (tindex c = 0; c < odata.cols(); c ++)
                                {
                                        const tscalar delta =
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();

                                        if (tcumulate)
                                        {
                                                odata(r, c) += delta;
                                        }
                                        else
                                        {
                                                odata(r, c) = delta;
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
                        for (tindex r = 0; r < odata.rows(); r ++)
                        {
                                for (tindex c = 0; c < odata.cols(); c ++)
                                {
                                        const tscalar delta = weight *
                                        kdata.cwiseProduct(idata.block(r, c, kdata.rows(), kdata.cols())).sum();

                                        if (tcumulate)
                                        {
                                                odata(r, c) += delta;
                                        }
                                        else
                                        {
                                                odata(r, c) = delta;
                                        }
                                }
                       }
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

