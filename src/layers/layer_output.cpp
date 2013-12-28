#include "layer_output.h"
#include "text.h"
#include "vectorizer.h"
#include "util/math.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        output_layer_t::output_layer_t(const string_t& params)
                :       m_params(params)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t output_layer_t::resize(size_t idims, size_t irows, size_t icols)
        {
                const size_t odims = math::clamp(text::from_params<size_t>(m_params, "odims", 10), 1, 256);
                const size_t orows = 1;
                const size_t ocols = 1;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_kdata.resize(odims, idims, irows, icols);
                m_gkdata.resize(odims, idims, irows, icols);

                m_bdata.resize(odims, 1, 1);
                m_gbdata.resize(odims, 1, 1);

                return m_kdata.size() + m_bdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void output_layer_t::zero_params()
        {
                m_kdata.zero();
                m_bdata.zero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void output_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_kdata.random(min, max);
                m_bdata.random(min, max);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& output_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_kdata << m_bdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& output_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gkdata << m_gbdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& output_layer_t::load_params(ivectorizer_t& s)
        {
                return s >> m_kdata >> m_bdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool output_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_params;
                oa << m_kdata;
                oa << m_bdata;

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool output_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_params;
                ia >> m_kdata;
                ia >> m_bdata;

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& output_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_idims() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                // convolution output
                m_idata = input;
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        matrix_t& odata = m_odata(o);

                        // bias
                        scalar_t& out = odata(0, 0);
                        out = bias(o);

                        // kernel
                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& kdata = m_kdata(o, i);

                                out += idata.cwiseProduct(kdata).sum();
                        }
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor3d_t& output_layer_t::backward(const tensor3d_t& gradient) const
        {
                assert(n_odims() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                // parameters gradient
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& gdata = gradient(o);
                        const scalar_t gout = gdata(0, 0);

                        // bias
                        gbias(o) = gout;

                        // kernel
                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                matrix_t& gkdata = m_gkdata(o, i);

                                gkdata = gout * idata;
                        }
                }

                // input gradient
                m_idata.zero();
                for (size_t o = 0; o < n_odims(); o ++)
                {
                        const matrix_t& odata = gradient(o);
                        const scalar_t gout = odata(0, 0);

                        for (size_t i = 0; i < n_idims(); i ++)
                        {
                                const matrix_t& kdata = m_kdata(o, i);
                                matrix_t& idata = m_idata(i);

                                idata += gout * kdata;
                        }
                }

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

