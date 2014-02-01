#include "layer_softmax_abs_pool.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        softmax_abs_pool_layer_t::softmax_abs_pool_layer_t(const string_t&)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t softmax_abs_pool_layer_t::resize(size_t idims, size_t irows, size_t icols)
        {
                const size_t odims = idims;
                const size_t orows = (irows + 1) / 2;
                const size_t ocols = (icols + 1) / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_wdata.resize(idims, irows, icols);
                m_sdata.resize(odims, orows, ocols);
                m_tdata.resize(odims, orows, ocols);

                return 0;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& softmax_abs_pool_layer_t::forward(const tensor3d_t& input) const
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

                        forward(idata, wdata, sdata, tdata, odata);
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& softmax_abs_pool_layer_t::backward(const tensor3d_t& gradient) const
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

                        backward(gdata, wdata, sdata, tdata, idata);
                }

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


