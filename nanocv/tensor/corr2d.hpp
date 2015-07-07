#pragma once

#include "matrix.hpp"

namespace ncv
{
        namespace tensor
        {
                ///
                /// \brief create the Toeplitz-like matrix to replace
                ///     the 2D convolution: idata += odata @ kdata
                ///     with the more eficient: as_vector(idata) = transpose(::operator()) * as_vector(kdata)
                ///
                template
                <
                        typename tscalar_,
                        typename tscalar = typename std::remove_const<tscalar_>::type,
                        typename tmatrix = typename tensor::matrix_types_t<tscalar>::tmatrix
                >
                struct corr2d_linearizer_t
                {
                        ///
                        /// \brief update buffer with new inputs
                        ///
                        template
                        <
                                typename tmatrixo,
                                typename tmatrixk,
                                typename tsize = typename tmatrixk::Index
                        >
                        const tmatrix& operator()(const tmatrixo& odata, const tmatrixk& kdata)
                        {
                                const tsize krows = kdata.rows();
                                const tsize kcols = kdata.cols();
                                const tsize ksize = krows * kcols;

                                const tsize orows = odata.rows();
                                const tsize ocols = odata.cols();

                                const tsize irows = orows + krows - 1;
                                const tsize icols = ocols + kcols - 1;
                                const tsize isize = irows * icols;

                                m_transf.resize(ksize, isize);
                                m_transf.setZero();

                                /// \todo more efficient construction
                                for (tsize r = 0; r < orows; r ++)
                                {
                                        for (tsize kr = 0; kr < krows; kr ++)
                                        {
                                                for (tsize c = 0; c < ocols; c ++)
                                                {
                                                        for (tsize kc = 0; kc < kcols; kc ++)
                                                        {
                                                                m_transf(kr * kcols + kc, (r + kr) * icols + (c + kc)) +=
                                                                odata(r, c);
                                                        }
                                                }
                                        }
                                }

                                return m_transf;
                        }

                        // attributes
                        tmatrix         m_transf;       ///< Linearized odata (buffered to reduce memory allocations)
                };
        }
}

