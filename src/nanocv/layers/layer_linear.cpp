#include "layer_linear.h"
#include "text.h"
#include "common/math.hpp"
#include "common/random.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(
                const tscalar* idata, tsize isize,
                const tscalar* wdata,
                const tscalar* bdata,
                tscalar* odata, tsize osize)
        {
                // output
                make_vector(odata, osize) =
                        make_vector(bdata, osize) +
                        make_matrix(wdata, osize, isize) *
                        make_vector(idata, isize);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(
                tscalar* idata, tsize isize,
                const tscalar* wdata, tscalar* gwdata,
                tscalar* gbdata,
                const tscalar* odata, tsize osize)
        {
                // bias & weights gradient
                make_vector(gbdata, osize) =
                        make_vector(odata, osize);

                make_matrix(gwdata, osize, isize) =
                        make_vector(odata, osize) *
                        make_vector(idata, isize).transpose();

                // input gradient
                make_vector(idata, isize) =
                        make_matrix(wdata, osize, isize).transpose() *
                        make_vector(odata, osize);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        linear_layer_t::linear_layer_t(const string_t& parameters)
                :       layer_t(parameters, "fully-connected linear layer, parameters: dims=10[1,4096]")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t linear_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.size();
                const size_t odims = math::clamp(text::from_params<size_t>(parameters(), "dims", 10), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(1, odims, idims);
                m_bdata.resize(odims, 1, 1);

                m_gwdata.resize(1, odims, idims);
                m_gbdata.resize(odims, 1, 1);

                return n_parameters();
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

                _forward(m_idata.data(), isize(),
                         m_wdata.data(), m_bdata.data(),
                         m_odata.data(), osize());

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::backward(const tensor_t& gradient)
        {
                assert(gradient.dims() == m_odata.dims());
                assert(gradient.rows() == m_odata.rows());
                assert(gradient.cols() == m_odata.cols());

                m_odata.copy_from(gradient);

                _backward(m_idata.data(), isize(),
                          m_wdata.data(), m_gwdata.data(),
                          m_gbdata.data(),
                          m_odata.data(), osize());

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

