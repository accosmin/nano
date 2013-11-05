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
                                typename tdotop          // column-based dot operator
                        >
                        void conv_add(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata,
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
                                                        podata[c] += dop(pidata + c, pkdata, kcols);
                                                }
                                        }
                                }
                        }
                }

                // 2D convolution: odata += idata * kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix
                >
                void conv_add_mod4(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv_add(idata, kdata, odata,
                                math::dot_mod4<typename tmatrix::Scalar>,
                                static_cast<int>(kdata.rows()), static_cast<int>(kdata.cols()));
                }

                // 2D convolution: odata += idata * kdata
                //      loop unrolling using 8 operations
                template
                <
                        typename tmatrix
                >
                void conv_add_mod8(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv_add(idata, kdata, odata,
                                math::dot_mod8<typename tmatrix::Scalar>,
                                static_cast<int>(kdata.rows()), static_cast<int>(kdata.cols()));
                }

                // 2D convolution: odata += idata * kdata
                //      no loop unrolling
                template
                <
                        typename tmatrix
                >
                void conv_add(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv_add(idata, kdata, odata,
                                math::dot<typename tmatrix::Scalar>,
                                static_cast<int>(kdata.rows()), static_cast<int>(kdata.cols()));
                }

                // 2D convolution: odata += idata * kdata
                //      for fixed size convolution (number of rows & columns)
                template
                <
                        int tkrows,
                        int tkcols,
                        typename tmatrix
                >
                void conv_add(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv_add(idata, kdata, odata,
                                math::dot<tkcols, typename tmatrix::Scalar>,
                                tkrows, tkcols);
                }

                // 2D convolution: odata += idata * kdata
                //      using Eigen blocks
                template
                <
                        typename tmatrix
                >
                void conv_add_eigen_block(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

                        const int orows = static_cast<int>(odata.rows());
                        const int ocols = static_cast<int>(odata.cols());

                        for (int r = 0; r < orows; r ++)
                        {
                                for (int c = 0; c < ocols; c ++)
                                {
                                        odata(r, c) += kdata.cwiseProduct(idata.block(r, c, krows, kcols)).sum();
                                }
                       }
                }

                // 2D convolution: odata += idata * kdata
                //      using run-time optimizations (fixed size for small convolutions, mod4 or mod8)
                template
                <
                        int tkrows,
                        typename tmatrix
                >
                void conv_add_dynamic(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kdata.cols());

                        if (kcols <= 32)
                        {
                                     if (kcols == 1 ) { conv_add<tkrows, 1 >(idata, kdata, odata); }
                                else if (kcols == 2 ) { conv_add<tkrows, 2 >(idata, kdata, odata); }
                                else if (kcols == 3 ) { conv_add<tkrows, 3 >(idata, kdata, odata); }
                                else if (kcols == 4 ) { conv_add<tkrows, 4 >(idata, kdata, odata); }
                                else if (kcols == 5 ) { conv_add<tkrows, 5 >(idata, kdata, odata); }
                                else if (kcols == 6 ) { conv_add<tkrows, 6 >(idata, kdata, odata); }
                                else if (kcols == 7 ) { conv_add<tkrows, 7 >(idata, kdata, odata); }
                                else if (kcols == 8 ) { conv_add<tkrows, 8 >(idata, kdata, odata); }
                                else if (kcols == 9 ) { conv_add<tkrows, 9 >(idata, kdata, odata); }
                                else if (kcols == 10) { conv_add<tkrows, 10>(idata, kdata, odata); }
                                else if (kcols == 11) { conv_add<tkrows, 11>(idata, kdata, odata); }
                                else if (kcols == 12) { conv_add<tkrows, 12>(idata, kdata, odata); }
                                else if (kcols == 13) { conv_add<tkrows, 13>(idata, kdata, odata); }
                                else if (kcols == 14) { conv_add<tkrows, 14>(idata, kdata, odata); }
                                else if (kcols == 15) { conv_add<tkrows, 15>(idata, kdata, odata); }
                                else if (kcols == 16) { conv_add<tkrows, 16>(idata, kdata, odata); }
                                else if (kcols == 17) { conv_add<tkrows, 17>(idata, kdata, odata); }
                                else if (kcols == 18) { conv_add<tkrows, 18>(idata, kdata, odata); }
                                else if (kcols == 19) { conv_add<tkrows, 19>(idata, kdata, odata); }
                                else if (kcols == 20) { conv_add<tkrows, 20>(idata, kdata, odata); }
                                else if (kcols == 21) { conv_add<tkrows, 21>(idata, kdata, odata); }
                                else if (kcols == 22) { conv_add<tkrows, 22>(idata, kdata, odata); }
                                else if (kcols == 23) { conv_add<tkrows, 23>(idata, kdata, odata); }
                                else if (kcols == 24) { conv_add<tkrows, 24>(idata, kdata, odata); }
                                else if (kcols == 25) { conv_add<tkrows, 25>(idata, kdata, odata); }
                                else if (kcols == 26) { conv_add<tkrows, 26>(idata, kdata, odata); }
                                else if (kcols == 27) { conv_add<tkrows, 27>(idata, kdata, odata); }
                                else if (kcols == 28) { conv_add<tkrows, 28>(idata, kdata, odata); }
                                else if (kcols == 29) { conv_add<tkrows, 29>(idata, kdata, odata); }
                                else if (kcols == 30) { conv_add<tkrows, 30>(idata, kdata, odata); }
                                else if (kcols == 31) { conv_add<tkrows, 31>(idata, kdata, odata); }
                                else if (kcols == 32) { conv_add<tkrows, 32>(idata, kdata, odata); }
                        }
                }
                template
                <
                        typename tmatrix
                >
                void conv_add_dynamic(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

                        if (krows <= 32 && kcols <= 32)
                        {
                                     if (krows == 1 ) { conv_add_dynamic<1 >(idata, kdata, odata); }
                                else if (krows == 2 ) { conv_add_dynamic<2 >(idata, kdata, odata); }
                                else if (krows == 3 ) { conv_add_dynamic<3 >(idata, kdata, odata); }
                                else if (krows == 4 ) { conv_add_dynamic<4 >(idata, kdata, odata); }
                                else if (krows == 5 ) { conv_add_dynamic<5 >(idata, kdata, odata); }
                                else if (krows == 6 ) { conv_add_dynamic<6 >(idata, kdata, odata); }
                                else if (krows == 7 ) { conv_add_dynamic<7 >(idata, kdata, odata); }
                                else if (krows == 8 ) { conv_add_dynamic<8 >(idata, kdata, odata); }
                                else if (krows == 9 ) { conv_add_dynamic<9 >(idata, kdata, odata); }
                                else if (krows == 10) { conv_add_dynamic<10>(idata, kdata, odata); }
                                else if (krows == 11) { conv_add_dynamic<11>(idata, kdata, odata); }
                                else if (krows == 12) { conv_add_dynamic<12>(idata, kdata, odata); }
                                else if (krows == 13) { conv_add_dynamic<13>(idata, kdata, odata); }
                                else if (krows == 14) { conv_add_dynamic<14>(idata, kdata, odata); }
                                else if (krows == 15) { conv_add_dynamic<15>(idata, kdata, odata); }
                                else if (krows == 16) { conv_add_dynamic<16>(idata, kdata, odata); }
                                else if (krows == 17) { conv_add_dynamic<17>(idata, kdata, odata); }
                                else if (krows == 18) { conv_add_dynamic<18>(idata, kdata, odata); }
                                else if (krows == 19) { conv_add_dynamic<19>(idata, kdata, odata); }
                                else if (krows == 20) { conv_add_dynamic<20>(idata, kdata, odata); }
                                else if (krows == 21) { conv_add_dynamic<21>(idata, kdata, odata); }
                                else if (krows == 22) { conv_add_dynamic<22>(idata, kdata, odata); }
                                else if (krows == 23) { conv_add_dynamic<23>(idata, kdata, odata); }
                                else if (krows == 24) { conv_add_dynamic<24>(idata, kdata, odata); }
                                else if (krows == 25) { conv_add_dynamic<25>(idata, kdata, odata); }
                                else if (krows == 26) { conv_add_dynamic<26>(idata, kdata, odata); }
                                else if (krows == 27) { conv_add_dynamic<27>(idata, kdata, odata); }
                                else if (krows == 28) { conv_add_dynamic<28>(idata, kdata, odata); }
                                else if (krows == 29) { conv_add_dynamic<29>(idata, kdata, odata); }
                                else if (krows == 30) { conv_add_dynamic<30>(idata, kdata, odata); }
                                else if (krows == 31) { conv_add_dynamic<31>(idata, kdata, odata); }
                                else if (krows == 32) { conv_add_dynamic<32>(idata, kdata, odata); }
                        }

                        else if (kcols % 8 == 0)
                        {
                                conv_add_mod8(idata, kdata, odata);
                        }

                        else
                        {
                                conv_add_mod4(idata, kdata, odata);
                        }
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

