#include "layer_linear.h"
#include "text.h"
#include "common/math.hpp"
#include "common/random.hpp"
#include "common/dot.hpp"
#include "common/thread_loop.hpp"

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
//                #pragma omp parallel for schedule(static)
//                for (tsize o = 0; o < osize; o ++)
//                {
//                        odata[o] = bdata[o] + math::dot_mod4x(wdata + o * isize, idata, isize);
//                }

                static ncv::thread_pool_t pool(ncv::n_threads() / 2);

                ncv::thread_loop(osize, [&] (size_t o)
                {
                        odata[o] = bdata[o] + math::dot_mod4x(wdata + o * isize, idata, isize);
                }, pool);
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

                // parameters gradient
                m_gwdata.copy_from(matrix_t(gradient.vector() * m_idata.vector().transpose()));
                m_gbdata.copy_from(gradient);

                // input gradient
                m_idata.copy_from(vector_t(m_wdata.plane_matrix(0).transpose() * gradient.vector()));

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

