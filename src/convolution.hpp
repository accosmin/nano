#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include "dot.hpp"

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

                // 2D convolution: odata += (weight *) idata @ kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv_mod4(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv(idata, kdata, odata, math::dot_mod4<tscalar>, kdata.rows(), kdata.cols());
                }

                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv_mod4(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv(idata, kdata, weight, odata, math::dot_mod4<tscalar>, kdata.rows(), kdata.cols());
                }

                // 2D convolution: odata += (weight *) idata @ kdata
                //      no loop unrolling
                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv(idata, kdata, odata, math::dot<tscalar>, kdata.rows(), kdata.cols());
                }

                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv(idata, kdata, weight, odata, math::dot<tscalar>, kdata.rows(), kdata.cols());
                }

                // 2D convolution: odata += (weight *) idata @ kdata
                //      for fixed size convolution (number of rows & columns)
                template
                <
                        int tkrows,
                        int tkcols,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv(idata, kdata, odata, math::dot<tkcols, tscalar>, tkrows, tkcols);
                }

                template
                <
                        int tkrows,
                        int tkcols,
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void wconv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::wconv(idata, kdata, weight, odata, math::dot<tkcols, tscalar>, tkrows, tkcols);
                }

                // 2D convolution: odata += (weight *) idata @ kdata
                //      using Eigen blocks
                template
                <
                        typename tmatrix,
                        typename tindex = typename tmatrix::Index
                >
                void conv_eigen(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
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

                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar,
                        typename tindex = typename tmatrix::Index
                >
                void wconv_eigen(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
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
        }
}

#endif // NANOCV_CONVOLUTION_H

