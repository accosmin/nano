#ifndef NANOCV_CONVOLUTION_H
#define NANOCV_CONVOLUTION_H

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
                                int tksize,
                                typename tscalar
                        >
                        tscalar conv_add(const tscalar* pidata, const tscalar* pkdata)
                        {
                                tscalar sum = 0;
                                for (int k = 0; k < tksize; k ++)
                                {
                                        sum += pidata[k] * pkdata[k];
                                }

                                return sum;
                        }

                        template
                        <
                                typename tscalar
                        >
                        tscalar conv_add(const tscalar* pidata, const tscalar* pkdata, int tksize)
                        {
                                tscalar sum = 0;
                                for (int k = 0; k < tksize; k ++)
                                {
                                        sum += pidata[k] * pkdata[k];
                                }

                                return sum;
                        }

                        template
                        <
                                typename tscalar
                        >
                        tscalar conv_add_mod4(const tscalar* pidata, const tscalar* pkdata, int tksize4, int tksize)
                        {
                                tscalar sum = 0;
                                for (int k = 0; k < tksize4; k += 4)
                                {
                                        sum += pidata[k + 0] * pkdata[k + 0];
                                        sum += pidata[k + 1] * pkdata[k + 1];
                                        sum += pidata[k + 2] * pkdata[k + 2];
                                        sum += pidata[k + 3] * pkdata[k + 3];
                                }
                                for (int k = tksize4; k < tksize; k ++)
                                {
                                        sum += pidata[k + 0] * pkdata[k + 0];
                                }

                                return sum;
                        }

                        template
                        <
                                typename tscalar
                        >
                        tscalar conv_add_mod8(const tscalar* pidata, const tscalar* pkdata, int tksize8, int tksize)
                        {
                                tscalar sum = 0;
                                for (int k = 0; k < tksize8; k += 8)
                                {
                                        sum += pidata[k + 0] * pkdata[k + 0];
                                        sum += pidata[k + 1] * pkdata[k + 1];
                                        sum += pidata[k + 2] * pkdata[k + 2];
                                        sum += pidata[k + 3] * pkdata[k + 3];
                                        sum += pidata[k + 4] * pkdata[k + 4];
                                        sum += pidata[k + 5] * pkdata[k + 5];
                                        sum += pidata[k + 6] * pkdata[k + 6];
                                        sum += pidata[k + 7] * pkdata[k + 7];
                                }
                                for (int k = tksize8; k < tksize; k ++)
                                {
                                        sum += pidata[k + 0] * pkdata[k + 0];
                                }

                                return sum;
                        }
                }

                // 2D (cumulative) convolution: odata += idata * kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix
                >
                void conv_add_mod4(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());
                        const int kcols4 = kcols - (kcols % 4);

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
                                                podata[c] += impl::conv_add_mod4(pidata + c, pkdata, kcols4, kcols);
                                        }
                                }                                
                        }
                }

                // 2D (cumulative) convolution: odata += idata * kdata
                //      loop unrolling using 8 operations
                template
                <
                        typename tmatrix
                >
                void conv_add_mod8(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());
                        const int kcols8 = kcols - (kcols % 8);

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
                                                podata[c] += impl::conv_add_mod8(pidata + c, pkdata, kcols8, kcols);
                                        }
                               }
                        }
                }

                // 2D (cumulative) convolution: odata += idata * kdata
                //      no loop unrolling
                template
                <
                        typename tmatrix
                >
                void conv_add_naive(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

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
                                                podata[c] += impl::conv_add(pidata + c, pkdata, kcols);
                                        }
                               }
                        }
                }

                // 2D (cumulative) convolution: odata += idata * kdata
                //      for fixed size convolution (number of rows & columns)
                template
                <
                        int tkrows,
                        int tkcols,
                        typename tmatrix
                >
                void conv_add_fixed(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int orows = static_cast<int>(odata.rows());
                        const int ocols = static_cast<int>(odata.cols());

                        for (int r = 0; r < orows; r ++)
                        {
                                typename tmatrix::Scalar* podata = &odata(r, 0);

                                for (int kr = 0; kr < tkrows; kr ++)
                                {
                                        const typename tmatrix::Scalar* pidata = &idata(r + kr, 0);
                                        const typename tmatrix::Scalar* pkdata = &kdata(kr, 0);

                                        for (int c = 0; c < ocols; c ++)
                                        {
                                                podata[c] += impl::conv_add(pidata + c, pkdata, tkcols);
                                        }
                               }
                        }
                }

                // 2D (cumulative) convolution: odata += idata * kdata
                //      for fixed size convolution (number of columns)
                template
                <
                        int tkcols,
                        typename tmatrix
                >
                void conv_add_fixed(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());

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
                                                podata[c] += impl::conv_add<tkcols>(pidata + c, pkdata);
                                        }
                                }
                        }
                }

                // 2D (cumulative) convolution: odata += idata * kdata
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

                // 2D (cumulative) convolution: odata += idata * kdata
                //      using run-time optimizations (fixed size for small convolutions, mod4 or mod8)
                template
                <
                        typename tmatrix
                >
                void conv_add_dynamic(const tmatrix& idata, const tmatrix& kdata, tmatrix& odata)
                {
                        const int krows = static_cast<int>(kdata.rows());
                        const int kcols = static_cast<int>(kdata.cols());

                        if (kcols >= 1 && kcols < 32)
                        {
                                if (krows == kcols)
                                {
                                             if (kcols == 1 ) { conv_add_fixed<1 , 1 >(idata, kdata, odata); }
                                        else if (kcols == 2 ) { conv_add_fixed<2 , 2 >(idata, kdata, odata); }
                                        else if (kcols == 3 ) { conv_add_fixed<3 , 3 >(idata, kdata, odata); }
                                        else if (kcols == 4 ) { conv_add_fixed<4 , 4 >(idata, kdata, odata); }
                                        else if (kcols == 5 ) { conv_add_fixed<5 , 5 >(idata, kdata, odata); }
                                        else if (kcols == 6 ) { conv_add_fixed<6 , 6 >(idata, kdata, odata); }
                                        else if (kcols == 7 ) { conv_add_fixed<7 , 7 >(idata, kdata, odata); }
                                        else if (kcols == 8 ) { conv_add_fixed<8 , 8 >(idata, kdata, odata); }
                                        else if (kcols == 9 ) { conv_add_fixed<9 , 9 >(idata, kdata, odata); }
                                        else if (kcols == 10) { conv_add_fixed<10, 10>(idata, kdata, odata); }
                                        else if (kcols == 11) { conv_add_fixed<11, 11>(idata, kdata, odata); }
                                        else if (kcols == 12) { conv_add_fixed<12, 12>(idata, kdata, odata); }
                                        else if (kcols == 13) { conv_add_fixed<13, 13>(idata, kdata, odata); }
                                        else if (kcols == 14) { conv_add_fixed<14, 14>(idata, kdata, odata); }
                                        else if (kcols == 15) { conv_add_fixed<15, 15>(idata, kdata, odata); }
                                        else if (kcols == 16) { conv_add_fixed<16, 16>(idata, kdata, odata); }
                                        else if (kcols == 17) { conv_add_fixed<17, 17>(idata, kdata, odata); }
                                        else if (kcols == 18) { conv_add_fixed<18, 18>(idata, kdata, odata); }
                                        else if (kcols == 19) { conv_add_fixed<19, 19>(idata, kdata, odata); }
                                        else if (kcols == 20) { conv_add_fixed<20, 20>(idata, kdata, odata); }
                                        else if (kcols == 21) { conv_add_fixed<21, 21>(idata, kdata, odata); }
                                        else if (kcols == 22) { conv_add_fixed<22, 22>(idata, kdata, odata); }
                                        else if (kcols == 23) { conv_add_fixed<23, 23>(idata, kdata, odata); }
                                        else if (kcols == 24) { conv_add_fixed<24, 24>(idata, kdata, odata); }
                                        else if (kcols == 25) { conv_add_fixed<25, 25>(idata, kdata, odata); }
                                        else if (kcols == 26) { conv_add_fixed<26, 26>(idata, kdata, odata); }
                                        else if (kcols == 27) { conv_add_fixed<27, 27>(idata, kdata, odata); }
                                        else if (kcols == 28) { conv_add_fixed<28, 28>(idata, kdata, odata); }
                                        else if (kcols == 29) { conv_add_fixed<29, 29>(idata, kdata, odata); }
                                        else if (kcols == 30) { conv_add_fixed<30, 30>(idata, kdata, odata); }
                                        else if (kcols == 31) { conv_add_fixed<31, 31>(idata, kdata, odata); }
                                        else if (kcols == 32) { conv_add_fixed<32, 32>(idata, kdata, odata); }
                                }

                                else
                                {
                                             if (kcols == 1 ) { conv_add_fixed<1 >(idata, kdata, odata); }
                                        else if (kcols == 2 ) { conv_add_fixed<2 >(idata, kdata, odata); }
                                        else if (kcols == 3 ) { conv_add_fixed<3 >(idata, kdata, odata); }
                                        else if (kcols == 4 ) { conv_add_fixed<4 >(idata, kdata, odata); }
                                        else if (kcols == 5 ) { conv_add_fixed<5 >(idata, kdata, odata); }
                                        else if (kcols == 6 ) { conv_add_fixed<6 >(idata, kdata, odata); }
                                        else if (kcols == 7 ) { conv_add_fixed<7 >(idata, kdata, odata); }
                                        else if (kcols == 8 ) { conv_add_fixed<8 >(idata, kdata, odata); }
                                        else if (kcols == 9 ) { conv_add_fixed<9 >(idata, kdata, odata); }
                                        else if (kcols == 10) { conv_add_fixed<10>(idata, kdata, odata); }
                                        else if (kcols == 11) { conv_add_fixed<11>(idata, kdata, odata); }
                                        else if (kcols == 12) { conv_add_fixed<12>(idata, kdata, odata); }
                                        else if (kcols == 13) { conv_add_fixed<13>(idata, kdata, odata); }
                                        else if (kcols == 14) { conv_add_fixed<14>(idata, kdata, odata); }
                                        else if (kcols == 15) { conv_add_fixed<15>(idata, kdata, odata); }
                                        else if (kcols == 16) { conv_add_fixed<16>(idata, kdata, odata); }
                                        else if (kcols == 17) { conv_add_fixed<17>(idata, kdata, odata); }
                                        else if (kcols == 18) { conv_add_fixed<18>(idata, kdata, odata); }
                                        else if (kcols == 19) { conv_add_fixed<19>(idata, kdata, odata); }
                                        else if (kcols == 20) { conv_add_fixed<20>(idata, kdata, odata); }
                                        else if (kcols == 21) { conv_add_fixed<21>(idata, kdata, odata); }
                                        else if (kcols == 22) { conv_add_fixed<22>(idata, kdata, odata); }
                                        else if (kcols == 23) { conv_add_fixed<23>(idata, kdata, odata); }
                                        else if (kcols == 24) { conv_add_fixed<24>(idata, kdata, odata); }
                                        else if (kcols == 25) { conv_add_fixed<25>(idata, kdata, odata); }
                                        else if (kcols == 26) { conv_add_fixed<26>(idata, kdata, odata); }
                                        else if (kcols == 27) { conv_add_fixed<27>(idata, kdata, odata); }
                                        else if (kcols == 28) { conv_add_fixed<28>(idata, kdata, odata); }
                                        else if (kcols == 29) { conv_add_fixed<29>(idata, kdata, odata); }
                                        else if (kcols == 30) { conv_add_fixed<30>(idata, kdata, odata); }
                                        else if (kcols == 31) { conv_add_fixed<31>(idata, kdata, odata); }
                                        else if (kcols == 32) { conv_add_fixed<32>(idata, kdata, odata); }
                                }
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

