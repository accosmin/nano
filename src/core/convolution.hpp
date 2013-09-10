#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

#include <functional>
#include "dot.hpp"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // 2D convolution utilities.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                using namespace std::placeholders;

                // implementation detail
                namespace impl
                {
                        template
                        <
                                typename tmatrix,
                                typename tdotop
                        >
                        void conv_by_col(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata,
                                const tdotop& op, int krows)
                        {
                                const int orows = static_cast<int>(odata.rows());
                                const int ocols = static_cast<int>(odata.cols());

                                for (int r = 0; r < orows; r ++)
                                {
                                        typename tmatrix::Scalar* podata = &odata(r, 0);

                                        for (int c = 0; c < ocols; c ++)
                                        {
                                                podata[c] = 0;
                                        }

                                        for (int kr = 0; kr < krows; kr ++)
                                        {
                                                const typename tmatrix::Scalar* pidata = &idata(r + kr, 0);
                                                const typename tmatrix::Scalar* pkdata = &kdata(kr, 0);

                                                for (int c = 0; c < ocols; c ++)
                                                {
                                                        podata[c] += op(pidata + c, pkdata);
                                                }
                                        }
                                }
                        }

                        template
                        <
                                typename tmatrix,
                                typename tvector,
                                typename tdotop
                        >
                        void sep_conv_by_col(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                                tmatrix& bdata, tmatrix& odata, const tdotop& op)
                        {
                                const int irows = static_cast<int>(idata.rows());

                                const int orows = static_cast<int>(odata.rows());
                                const int ocols = static_cast<int>(odata.cols());

                                const int krows = static_cast<int>(krdata.size());

                                for (int r = 0; r < irows; r ++)
                                {
                                        typename tmatrix::Scalar* pbdata = &bdata(r, 0);

                                        for (int c = 0; c < ocols; c ++)
                                        {
                                                const typename tmatrix::Scalar* pidata = &idata(r, 0);
                                                const typename tvector::Scalar* pkdata = &kcdata(0);

                                                pbdata[c] = op(pidata + c, pkdata);
                                        }
                                }

                                odata.setZero();

                                for (int r = 0; r < orows; r ++)
                                {
                                        for (int kr = 0; kr < krows; kr ++)
                                        {
                                                const typename tmatrix::Scalar* pbdata = &bdata(r + kr, 0);
                                                const typename tvector::Scalar kdata = krdata(kr);
                                                typename tmatrix::Scalar* podata = &odata(r, 0);

                                                for (int c = 0; c < ocols; c ++)
                                                {
                                                        podata[c] += pbdata[c] * kdata;
                                                }
                                        }
                                }

//                                for (int c = 0; c < ocols; c ++)
//                                {
//                                        for (int r = 0; r < orows; r ++)
//                                        {
//                                                typename tmatrix::Scalar sum = 0;
//                                                for (int kr = 0; kr < krows; kr ++)
//                                                {
//                                                        sum += bdata(r + kr, c) * krdata(kr);
//                                                }

//                                                odata(r, c) = sum;
//                                        }
//                                }


//                                for (int c = 0; c < ocols; c ++)
//                                {
//                                        for (int r = 0; r < orows; r ++)
//                                        {
//                                                typename tmatrix::Scalar* podata = &odata(r, 0);

//                                                const typename tmatrix::Scalar* pidata = &idata(r, 0);
//                                                const typename tvector::Scalar* pkdata = &kcdata(0);

//                                                podata[c] += op(pidata + c, pkdata);
//                                        }
//                                }


//                                        for (int kr = 0; kr < krows; kr ++)
//                                        {
//                                                const typename tmatrix::Scalar* pidata = &idata(r + kr, 0);
//                                                const typename tmatrix::Scalar* pkdata = &kdata(kr, 0);

//                                                for (int c = 0; c < ocols; c ++)
//                                                {
//                                                        podata[c] += op(pidata + c, pkdata);
//                                                }
//                                        }
//                                }
                        }

                        template
                        <
                                typename tmatrix,
                                typename tdotop
                        >
                        void conv_by_col(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata,
                                const tdotop& op)
                        {
                                conv_by_col(idata, kdata, odata, op, static_cast<int>(kdata.rows()));
                        }

                        template
                        <
                                int tkrows,
                                typename tmatrix,
                                typename tdotop
                        >
                        void conv_by_col(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata,
                                const tdotop& op)
                        {
                                conv_by_col(idata, kdata, odata, op, tkrows);
                        }
                }

                // 2D convolution: odata += idata * kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix
                >
                void conv_mod4(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kdata.cols());
                        const int kcols4 = kcols - (kcols % 4);

                        impl::conv_by_col(idata, kdata, odata,
                                std::bind(math::dot_mod4<typename tmatrix::Scalar>, _1, _2, kcols4, kcols));
                }

                // separable 2D convolution: odata += idata * kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv_mod4(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kcdata.size());
                        const int kcols4 = kcols - (kcols % 4);

                        impl::sep_conv_by_col(idata, krdata, kcdata, bdata, odata,
                                std::bind(math::dot_mod4<typename tmatrix::Scalar>, _1, _2, kcols4, kcols));
                }

                // 2D convolution: odata += idata * kdata
                //      loop unrolling using 8 operations
                template
                <
                        typename tmatrix
                >
                void conv_mod8(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kdata.cols());
                        const int kcols8 = kcols - (kcols % 8);

                        impl::conv_by_col(idata, kdata, odata,
                                std::bind(math::dot_mod8<typename tmatrix::Scalar>, _1, _2, kcols8, kcols));
                }

                // separable 2D convolution: odata += idata * kdata
                //      loop unrolling using 8 operations
                template
                <
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv_mod8(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kcdata.size());
                        const int kcols8 = kcols - (kcols % 8);

                        impl::sep_conv_by_col(idata, krdata, kcdata, bdata, odata,
                                std::bind(math::dot_mod8<typename tmatrix::Scalar>, _1, _2, kcols8, kcols));
                }

                // 2D convolution: odata += idata * kdata
                //      no loop unrolling
                template
                <
                        typename tmatrix
                >
                void conv_brut(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kdata.cols());

                        impl::conv_by_col(idata, kdata, odata,
                                std::bind(math::dot<typename tmatrix::Scalar>, _1, _2, kcols));
                }

                // 2D convolution: odata += idata * kdata
                //      for fixed size convolution (number of rows & columns)
                template
                <
                        int tkrows,
                        int tkcols,
                        typename tmatrix
                >
                void conv_fixed(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv_by_col<tkrows>(idata, kdata, odata,
                                std::bind(math::dot<tkcols, typename tmatrix::Scalar>, _1, _2));
                }

                // 2D convolution: odata += idata * kdata
                //      for fixed size convolution (number of columns)
                template
                <
                        int tkcols,
                        typename tmatrix
                >
                void conv_fixed(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        impl::conv_by_col(idata, kdata, odata,
                                std::bind(math::dot<tkcols, typename tmatrix::Scalar>, _1, _2));
                }

                // 2D convolution: odata += idata * kdata
                //      using Eigen blocks
                template
                <
                        typename tmatrix
                >
                void conv_eigen_block(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

                        const int orows = static_cast<int>(odata.rows());
                        const int ocols = static_cast<int>(odata.cols());

                        for (int r = 0; r < orows; r ++)
                        {
                                for (int c = 0; c < ocols; c ++)
                                {
                                        odata(r, c) = kdata.cwiseProduct(idata.block(r, c, krows, kcols)).sum();
                                }
                       }
                }

                // 2D convolution: odata += idata * kdata
                //      using run-time optimizations (fixed size for small convolutions, mod4 or mod8)
                template
                <
                        typename tmatrix
                >
                void conv_dynamic(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

                        if (kcols >= 1 && kcols < 32)
                        {
                                if (krows == kcols)
                                {
                                             if (kcols == 1 ) { conv_fixed<1 , 1 >(idata, kdata, odata); }
                                        else if (kcols == 2 ) { conv_fixed<2 , 2 >(idata, kdata, odata); }
                                        else if (kcols == 3 ) { conv_fixed<3 , 3 >(idata, kdata, odata); }
                                        else if (kcols == 4 ) { conv_fixed<4 , 4 >(idata, kdata, odata); }
                                        else if (kcols == 5 ) { conv_fixed<5 , 5 >(idata, kdata, odata); }
                                        else if (kcols == 6 ) { conv_fixed<6 , 6 >(idata, kdata, odata); }
                                        else if (kcols == 7 ) { conv_fixed<7 , 7 >(idata, kdata, odata); }
                                        else if (kcols == 8 ) { conv_fixed<8 , 8 >(idata, kdata, odata); }
                                        else if (kcols == 9 ) { conv_fixed<9 , 9 >(idata, kdata, odata); }
                                        else if (kcols == 10) { conv_fixed<10, 10>(idata, kdata, odata); }
                                        else if (kcols == 11) { conv_fixed<11, 11>(idata, kdata, odata); }
                                        else if (kcols == 12) { conv_fixed<12, 12>(idata, kdata, odata); }
                                        else if (kcols == 13) { conv_fixed<13, 13>(idata, kdata, odata); }
                                        else if (kcols == 14) { conv_fixed<14, 14>(idata, kdata, odata); }
                                        else if (kcols == 15) { conv_fixed<15, 15>(idata, kdata, odata); }
                                        else if (kcols == 16) { conv_fixed<16, 16>(idata, kdata, odata); }
                                        else if (kcols == 17) { conv_fixed<17, 17>(idata, kdata, odata); }
                                        else if (kcols == 18) { conv_fixed<18, 18>(idata, kdata, odata); }
                                        else if (kcols == 19) { conv_fixed<19, 19>(idata, kdata, odata); }
                                        else if (kcols == 20) { conv_fixed<20, 20>(idata, kdata, odata); }
                                        else if (kcols == 21) { conv_fixed<21, 21>(idata, kdata, odata); }
                                        else if (kcols == 22) { conv_fixed<22, 22>(idata, kdata, odata); }
                                        else if (kcols == 23) { conv_fixed<23, 23>(idata, kdata, odata); }
                                        else if (kcols == 24) { conv_fixed<24, 24>(idata, kdata, odata); }
                                        else if (kcols == 25) { conv_fixed<25, 25>(idata, kdata, odata); }
                                        else if (kcols == 26) { conv_fixed<26, 26>(idata, kdata, odata); }
                                        else if (kcols == 27) { conv_fixed<27, 27>(idata, kdata, odata); }
                                        else if (kcols == 28) { conv_fixed<28, 28>(idata, kdata, odata); }
                                        else if (kcols == 29) { conv_fixed<29, 29>(idata, kdata, odata); }
                                        else if (kcols == 30) { conv_fixed<30, 30>(idata, kdata, odata); }
                                        else if (kcols == 31) { conv_fixed<31, 31>(idata, kdata, odata); }
                                        else if (kcols == 32) { conv_fixed<32, 32>(idata, kdata, odata); }
                                }

                                else
                                {
                                             if (kcols == 1 ) { conv_fixed<1 >(idata, kdata, odata); }
                                        else if (kcols == 2 ) { conv_fixed<2 >(idata, kdata, odata); }
                                        else if (kcols == 3 ) { conv_fixed<3 >(idata, kdata, odata); }
                                        else if (kcols == 4 ) { conv_fixed<4 >(idata, kdata, odata); }
                                        else if (kcols == 5 ) { conv_fixed<5 >(idata, kdata, odata); }
                                        else if (kcols == 6 ) { conv_fixed<6 >(idata, kdata, odata); }
                                        else if (kcols == 7 ) { conv_fixed<7 >(idata, kdata, odata); }
                                        else if (kcols == 8 ) { conv_fixed<8 >(idata, kdata, odata); }
                                        else if (kcols == 9 ) { conv_fixed<9 >(idata, kdata, odata); }
                                        else if (kcols == 10) { conv_fixed<10>(idata, kdata, odata); }
                                        else if (kcols == 11) { conv_fixed<11>(idata, kdata, odata); }
                                        else if (kcols == 12) { conv_fixed<12>(idata, kdata, odata); }
                                        else if (kcols == 13) { conv_fixed<13>(idata, kdata, odata); }
                                        else if (kcols == 14) { conv_fixed<14>(idata, kdata, odata); }
                                        else if (kcols == 15) { conv_fixed<15>(idata, kdata, odata); }
                                        else if (kcols == 16) { conv_fixed<16>(idata, kdata, odata); }
                                        else if (kcols == 17) { conv_fixed<17>(idata, kdata, odata); }
                                        else if (kcols == 18) { conv_fixed<18>(idata, kdata, odata); }
                                        else if (kcols == 19) { conv_fixed<19>(idata, kdata, odata); }
                                        else if (kcols == 20) { conv_fixed<20>(idata, kdata, odata); }
                                        else if (kcols == 21) { conv_fixed<21>(idata, kdata, odata); }
                                        else if (kcols == 22) { conv_fixed<22>(idata, kdata, odata); }
                                        else if (kcols == 23) { conv_fixed<23>(idata, kdata, odata); }
                                        else if (kcols == 24) { conv_fixed<24>(idata, kdata, odata); }
                                        else if (kcols == 25) { conv_fixed<25>(idata, kdata, odata); }
                                        else if (kcols == 26) { conv_fixed<26>(idata, kdata, odata); }
                                        else if (kcols == 27) { conv_fixed<27>(idata, kdata, odata); }
                                        else if (kcols == 28) { conv_fixed<28>(idata, kdata, odata); }
                                        else if (kcols == 29) { conv_fixed<29>(idata, kdata, odata); }
                                        else if (kcols == 30) { conv_fixed<30>(idata, kdata, odata); }
                                        else if (kcols == 31) { conv_fixed<31>(idata, kdata, odata); }
                                        else if (kcols == 32) { conv_fixed<32>(idata, kdata, odata); }
                                }
                        }

                        else if (kcols % 8 == 0)
                        {
                                conv_mod8(idata, kdata, odata);
                        }

                        else
                        {
                                conv_mod4(idata, kdata, odata);
                        }
                }
        }
}

#endif // NANOCV_CONVOLUTION_H

