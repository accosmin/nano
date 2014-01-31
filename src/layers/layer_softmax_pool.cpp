#include "layer_softmax_pool.h"
#include "util/math.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        softmax_pool_layer_t::softmax_pool_layer_t(const string_t&)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t softmax_pool_layer_t::resize(size_t idims, size_t irows, size_t icols)
        {
                const size_t odims = idims;
                const size_t orows = irows / 2;
                const size_t ocols = icols / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_wdata.resize(idims, irows, icols);
                m_sdata.resize(odims, orows, ocols);
                m_tdata.resize(odims, orows, ocols);

                return 0;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tmatrix,
                typename tscalar = typename tmatrix::Scalar
        >
        void _forward(const tmatrix& idata, tmatrix& wdata, tmatrix& sdata, tmatrix& tdata, tmatrix& odata)
        {
                wdata = idata.array().exp().matrix();

                sdata.setZero();
                tdata.setZero();

                for (auto r = 0, rr = 0; r < idata.rows(); r ++, rr = r / 2)
                {
                        for (auto c = 0, cc = 0; c < idata.cols(); c ++, cc = c / 2)
                        {
                                sdata(rr, cc) += wdata(r, c) * idata(r, c);
                                tdata(rr, cc) += wdata(r, c);
                        }
                }

                odata = (sdata.array() / tdata.array()).matrix();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& softmax_pool_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_idims() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                m_idata = input;

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& idata = m_idata(o);
                        matrix_t& wdata = m_wdata(o);
                        matrix_t& sdata = m_sdata(o);
                        matrix_t& tdata = m_tdata(o);
                        matrix_t& odata = m_odata(o);

                        _forward(idata, wdata, sdata, tdata, odata);
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tmatrix,
                typename tscalar = typename tmatrix::Scalar
        >
        void _backward(
                const tmatrix& gdata, const tmatrix& wdata, const tmatrix& sdata, const tmatrix& tdata, tmatrix& idata)
        {
                for (auto r = 0, rr = 0; r < idata.rows(); r ++, rr = r / 2)
                {
                        for (auto c = 0, cc = 0; c < idata.cols(); c ++, cc = c / 2)
                        {
                                const auto wv = wdata(r, c);
                                const auto sv = sdata(rr, cc);
                                const auto tv = tdata(rr, cc);

                                idata(r, c) =   gdata(rr, cc) *
                                                (tv * (wv + wv * idata(r, c)) - sv * wv) / math::square(tv);
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& softmax_pool_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_odims() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& gdata = gradient(o);
                        const matrix_t& wdata = m_wdata(o);
                        const matrix_t& sdata = m_sdata(o);
                        const matrix_t& tdata = m_tdata(o);
                        matrix_t& idata = m_idata(o);

                        _backward(gdata, wdata, sdata, tdata, idata);
                }

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


