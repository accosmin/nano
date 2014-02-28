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

                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(1, odims, idims);
                m_bdata.resize(odims, 1, 1);

                m_gwdata.resize(1, odims, idims);
                m_gbdata.resize(odims, 1, 1);

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

        const tensor_t& linear_layer_t::forward(const tensor_t& input)
        {
                assert(input.dims() == m_idata.dims());
                assert(input.rows() == m_idata.rows());
                assert(input.cols() == m_idata.cols());

                m_idata.copy_from(input);
                m_odata.copy_from(vector_t(m_bdata.vector() + m_wdata.plane_matrix(0) * m_idata.vector()));

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::backward(const tensor_t& gradient)
        {
                assert(gradient.dims() == m_odata.dims());
                assert(gradient.rows() == m_odata.rows());
                assert(gradient.cols() == m_odata.cols());

                // parameters gradient
                m_gwdata.copy_from(matrix_t(gradient.vector() * m_idata.vector().transpose()));
                m_gbdata.copy_from(gradient);

                // input gradient
                m_idata.copy_from(vector_t(m_wdata.plane_matrix(0).transpose() * gradient.vector()));

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

