#include "layer_linear.h"
#include "text.h"
#include "common/math.hpp"
#include "common/random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        linear_layer_t::linear_layer_t(const string_t& params)
                :       m_params(params)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t linear_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.size();
                const size_t odims = math::clamp(text::from_params<size_t>(m_params, "dims", 10), 1, 1024);

                m_idata.resize(idims, 1, 1, 1);
                m_odata.resize(odims, 1, 1, 1);

                m_wdata.resize(odims, idims, 1, 1);
                m_bdata.resize(odims, 1, 1, 1);

                m_gwdata.resize(odims, idims, 1, 1);
                m_gbdata.resize(odims, 1, 1, 1);

                return m_wdata.size() + m_bdata.size();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void linear_layer_t::zero_params()
        {
                m_wdata.zero();
                m_bdata.zero();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        void linear_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_wdata.random(random_t<scalar_t>(min, max));
                m_bdata.random(random_t<scalar_t>(min, max));
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& linear_layer_t::save_params(ovectorizer_t& s) const
        {
                return s << m_wdata << m_bdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ovectorizer_t& linear_layer_t::save_grad(ovectorizer_t& s) const
        {
                return s << m_gwdata << m_gbdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        ivectorizer_t& linear_layer_t::load_params(ivectorizer_t& s)
        {
                return s >> m_wdata >> m_bdata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool linear_layer_t::save(boost::archive::binary_oarchive& oa) const
        {
                oa << m_params;
                oa << m_wdata;
                oa << m_bdata;

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        bool linear_layer_t::load(boost::archive::binary_iarchive& ia)
        {
                ia >> m_params;
                ia >> m_wdata;
                ia >> m_bdata;

                return true;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::forward(const tensor_t& input)
        {
                assert(isize() == m_idata.size());

                m_idata.copy_from(input);
                m_odata.vector() = m_bdata.vector() + m_wdata.matrix() * m_idata.vector();

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::backward(const tensor_t& gradient)
        {
                assert(osize() == gradient.size());

                // parameters gradient
                m_gwdata.matrix() = gradient.vector() * m_idata.vector().transpose();
                m_gbdata.copy_from(gradient);

                // input gradient
                m_idata.vector() = m_wdata.matrix().transpose() * gradient.vector();

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

