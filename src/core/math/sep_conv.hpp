#ifndef NANOCV_SEP_CONVOLUTION_H
#define NANOCV_SEP_CONVOLUTION_H

#include "dot.hpp"
#include "mad.hpp"

namespace ncv
{
        ////////////////////////////////////////////////////////////////////////////////////////////////////////
        // separable 2D convolution utilities.
        ////////////////////////////////////////////////////////////////////////////////////////////////////////

        namespace math
        {
                // implementation detail
                namespace impl
                {
                        template
                        <
                                typename tmatrix,
                                typename tvector,
                                typename tdotop,        // column-based dot operator
                                typename tmadop         // row-based mad operator
                        >
                        void sep_conv(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                                tmatrix& bdata, tmatrix& odata, const tdotop& dop, const tmadop& mop,
                                int kcols, int ocols)
                        {
                                const int irows = static_cast<int>(idata.rows());
                                const int orows = static_cast<int>(odata.rows());
                                const int krows = static_cast<int>(krdata.size());

                                for (int r = 0; r < irows; r ++)
                                {
                                        typename tmatrix::Scalar* pbdata = &bdata(r, 0);

                                        for (int c = 0; c < ocols; c ++)
                                        {
                                                const typename tmatrix::Scalar* pidata = &idata(r, 0);
                                                const typename tvector::Scalar* pkdata = &kcdata(0);

                                                pbdata[c] = dop(pidata + c, pkdata, kcols);
                                        }
                                }

                                odata.setZero();
                                for (int r = 0; r < orows; r ++)
                                {
                                        typename tmatrix::Scalar* podata = &odata(r, 0);

                                        for (int kr = 0; kr < krows; kr ++)
                                        {
                                                const typename tmatrix::Scalar* pbdata = &bdata(r + kr, 0);
                                                const typename tvector::Scalar kdata = krdata(kr);                                                

                                                mop(podata, kdata, pbdata, ocols);
                                        }
                                }
                        }
                }

                // separable 2D convolution: odata = idata * kdata
                //      loop unrolling using 4 operations
                template
                <
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv_mod4(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        impl::sep_conv(idata, krdata, kcdata, bdata, odata,
                                math::dot_mod4<typename tmatrix::Scalar>,
                                math::mad_mod4<typename tmatrix::Scalar>,
                                static_cast<int>(kcdata.size()),
                                static_cast<int>(odata.cols()));
                }

                // separable 2D convolution: odata = idata * kdata
                //      loop unrolling using 8 operations
                template
                <
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv_mod8(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        impl::sep_conv(idata, krdata, kcdata, bdata, odata,
                               math::dot_mod8<typename tmatrix::Scalar>,
                               math::mad_mod8<typename tmatrix::Scalar>,
                               static_cast<int>(kcdata.size()),
                               static_cast<int>(odata.cols()));
                }

                // separable 2D convolution: odata = idata * kdata
                //      no loop unrolling
                template
                <
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        impl::sep_conv(idata, krdata, kcdata, bdata, odata,
                               math::dot<typename tmatrix::Scalar>,
                               math::mad<typename tmatrix::Scalar>,
                               static_cast<int>(kcdata.size()),
                               static_cast<int>(odata.cols()));
                }

                // separable 2D convolution: odata = idata * kdata
                //      for fixed size convolution (number of columns)
                template
                <
                        int tkcols,
                        int tocols,
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        impl::sep_conv(idata, krdata, kcdata, bdata, odata,
                               math::dot<tkcols, typename tmatrix::Scalar>,
                               math::mad<tocols, typename tmatrix::Scalar>,
                               static_cast<int>(kcdata.size()),
                               static_cast<int>(odata.cols()));
                }

                // separable 2D convolution: odata = idata * kdata
                //      using run-time optimizations (fixed size for small convolutions, mod4 or mod8)
                template
                <
                        int tkcols,
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv_dynamic(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        const int ocols = static_cast<int>(odata.cols());

                        if (ocols <= 32)
                        {
                                     if (ocols == 1 ) { sep_conv<tkcols, 1 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 2 ) { sep_conv<tkcols, 2 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 3 ) { sep_conv<tkcols, 3 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 4 ) { sep_conv<tkcols, 4 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 5 ) { sep_conv<tkcols, 5 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 6 ) { sep_conv<tkcols, 6 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 7 ) { sep_conv<tkcols, 7 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 8 ) { sep_conv<tkcols, 8 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 9 ) { sep_conv<tkcols, 9 >(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 10) { sep_conv<tkcols, 10>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 11) { sep_conv<tkcols, 11>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 12) { sep_conv<tkcols, 12>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 13) { sep_conv<tkcols, 13>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 14) { sep_conv<tkcols, 14>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 15) { sep_conv<tkcols, 15>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 16) { sep_conv<tkcols, 16>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 17) { sep_conv<tkcols, 17>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 18) { sep_conv<tkcols, 18>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 19) { sep_conv<tkcols, 19>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 20) { sep_conv<tkcols, 20>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 21) { sep_conv<tkcols, 21>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 22) { sep_conv<tkcols, 22>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 23) { sep_conv<tkcols, 23>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 24) { sep_conv<tkcols, 24>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 25) { sep_conv<tkcols, 25>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 26) { sep_conv<tkcols, 26>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 27) { sep_conv<tkcols, 27>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 28) { sep_conv<tkcols, 28>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 29) { sep_conv<tkcols, 29>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 30) { sep_conv<tkcols, 30>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 31) { sep_conv<tkcols, 31>(idata, krdata, kcdata, bdata, odata); }
                                else if (ocols == 32) { sep_conv<tkcols, 32>(idata, krdata, kcdata, bdata, odata); }
                        }
                }
                template
                <
                        typename tmatrix,
                        typename tvector
                >
                void sep_conv_dynamic(const tmatrix& idata, const tvector& krdata, const tvector& kcdata,
                        tmatrix& bdata, tmatrix& odata)
                {
                        const int kcols = static_cast<int>(kcdata.size());
                        const int ocols = static_cast<int>(odata.cols());

                        if (kcols <= 32 && ocols <= 32)
                        {
                                     if (kcols == 1 ) { sep_conv_dynamic<1 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 2 ) { sep_conv_dynamic<2 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 3 ) { sep_conv_dynamic<3 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 4 ) { sep_conv_dynamic<4 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 5 ) { sep_conv_dynamic<5 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 6 ) { sep_conv_dynamic<6 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 7 ) { sep_conv_dynamic<7 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 8 ) { sep_conv_dynamic<8 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 9 ) { sep_conv_dynamic<9 >(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 10) { sep_conv_dynamic<10>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 11) { sep_conv_dynamic<11>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 12) { sep_conv_dynamic<12>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 13) { sep_conv_dynamic<13>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 14) { sep_conv_dynamic<14>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 15) { sep_conv_dynamic<15>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 16) { sep_conv_dynamic<16>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 17) { sep_conv_dynamic<17>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 18) { sep_conv_dynamic<18>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 19) { sep_conv_dynamic<19>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 20) { sep_conv_dynamic<20>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 21) { sep_conv_dynamic<21>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 22) { sep_conv_dynamic<22>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 23) { sep_conv_dynamic<23>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 24) { sep_conv_dynamic<24>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 25) { sep_conv_dynamic<25>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 26) { sep_conv_dynamic<26>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 27) { sep_conv_dynamic<27>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 28) { sep_conv_dynamic<28>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 29) { sep_conv_dynamic<29>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 30) { sep_conv_dynamic<30>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 31) { sep_conv_dynamic<31>(idata, krdata, kcdata, bdata, odata); }
                                else if (kcols == 32) { sep_conv_dynamic<32>(idata, krdata, kcdata, bdata, odata); }
                        }

                        else if (kcols % 8 == 0)
                        {
                                sep_conv_mod8(idata, krdata, kcdata, bdata, odata);
                        }

                        else
                        {
                                sep_conv_mod4(idata, krdata, kcdata, bdata, odata);
                        }
                }
        }
}

#endif // NANOCV_SEP_CONVOLUTION_H

