#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include "dot.hpp"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 2D convolution utilities.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // implementation detail
                namespace impl
                {
                        template
                        <
                                typename tmatrix,
                                typename tscalar,
                                typename tdotop          // column-based dot operator
                        >
                        void conv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata,
                                  const tdotop& dop, int krows, int kcols)
                        {
                                const int orows = static_cast<int>(odata.rows());
                                const int ocols = static_cast<int>(odata.cols());

                                for (int r = 0; r < orows; r ++)
                                {
                                        typename tmatrix::Scalar* podata = &odata(r, 0);

                                        for (int kr = 0; kr < krows; kr ++)
                                        {
                                                const typename tmatrix::Scalar* pidata = &idata(r + kr, 0);
                                                const typename tmatrix::Scalar* pkdata = &kdata(kr, 0);

                                                for (int c = 0; c < ocols; c ++)
                                                {
                                                        podata[c] += weight * dop(pidata + c, pkdata, kcols);
                                                }
                                        }
                                }
                        }
                }

                // 2D convolution: odata = idata * kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix,
                        typename tscalar
                >
                void conv_mod4(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::conv(idata, kdata, weight, odata,
                                math::dot_mod4<typename tmatrix::Scalar>,
                                static_cast<int>(kdata.rows()), static_cast<int>(kdata.cols()));
                }

                // 2D convolution: odata = idata * kdata
                //      loop unrolling using 8 operations
                template
                <
                        typename tmatrix,
                        typename tscalar
                >
                void conv_mod8(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::conv(idata, kdata, weight, odata,
                                math::dot_mod8<typename tmatrix::Scalar>,
                                static_cast<int>(kdata.rows()), static_cast<int>(kdata.cols()));
                }

                // 2D convolution: odata = idata * kdata
                //      no loop unrolling
                template
                <
                        typename tmatrix,
                        typename tscalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::conv(idata, kdata, weight, odata,
                                math::dot<typename tmatrix::Scalar>,
                                static_cast<int>(kdata.rows()), static_cast<int>(kdata.cols()));
                }

                // 2D convolution: odata = idata * kdata
                //      for fixed size convolution (number of rows & columns)
                template
                <
                        int tkrows,
                        int tkcols,
                        typename tmatrix,
                        typename tscalar
                >
                void conv(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        impl::conv(idata, kdata, weight, odata,
                                math::dot<tkcols, typename tmatrix::Scalar>,
                                tkrows, tkcols);
                }

                // 2D convolution: odata = idata * kdata
                //      using Eigen blocks
                template
                <
                        typename tmatrix,
                        typename tscalar
                >
                void conv_eigen(const tmatrix& idata, const tmatrix& kdata, tscalar weight, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

                        const int orows = static_cast<int>(odata.rows());
                        const int ocols = static_cast<int>(odata.cols());

                        for (int r = 0; r < orows; r ++)
                        {
                                for (int c = 0; c < ocols; c ++)
                                {
                                        odata(r, c) += weight * kdata.cwiseProduct(idata.block(r, c, krows, kcols)).sum();
                                }
                       }
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

