#pragma once

#include "matrix.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create the Toeplitz-like matrix to replace
                ///     the 2D convolution: odata += idata @ kdata
                ///     with the more efficient: as_vector(odata) = transpose(::operator()) * as_vector(kdata)
                ///
                template
                <
                        typename tscalar_,
                        typename tscalar = typename std::remove_const<tscalar_>::type,
                        typename tmatrix = typename tensor::matrix_types_t<tscalar>::tmatrix
                >
                struct conv2d_linearizer_t
                {
                        ///
                        /// \brief update buffer with new inputs
                        ///
                        template
                        <
                                typename tmatrixi,
                                typename tmatrixk,
                                typename tsize = typename tmatrixk::Index
                        >
                        const tmatrix& operator()(const tmatrixi& idata, const tmatrixk& kdata)
                        {
                                const tsize krows = kdata.rows();
                                const tsize kcols = kdata.cols();
                                const tsize ksize = krows * kcols;

                                const tsize orows = idata.rows() - krows + 1;
                                const tsize ocols = idata.cols() - kcols + 1;
                                const tsize osize = orows * ocols;

                                m_transf.resize(ksize, osize);

                                /// \todo more efficient construction
                                for (tsize r = 0; r < orows; r ++)
                                {
                                        for (tsize kr = 0; kr < krows; kr ++)
                                        {
                                                for (tsize c = 0; c < ocols; c ++)
                                                {
                                                        for (tsize kc = 0; kc < kcols; kc ++)
                                                        {
                                                                m_transf(kr * kcols + kc, r * ocols + c) =
                                                                idata(r + kr, c + kc);
                                                        }
                                                }
                                        }
                                }

                                return m_transf;
                        }

                        // attributes
                        tmatrix         m_transf;    ///< Linearized idata (buffered to reduce memory allocations)
                };
        }
}

