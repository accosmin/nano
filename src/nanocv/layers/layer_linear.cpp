#include "layer_linear.h"
#include "common/math.hpp"
#include "common/random.hpp"
#include "tensor/util.hpp"

namespace ncv
{
        template
        <
                typename tscalar,
                typename tsize
        >
        static void _output(
                const tscalar* idata, tsize isize,
                const tscalar* wdata,
                const tscalar* bdata,
                tscalar* odata, tsize osize)
        {
                // output
                tensor::make_vector(odata, osize).noalias() =
                        tensor::make_vector(bdata, osize) +
                        tensor::make_matrix(wdata, osize, isize) *
                        tensor::make_vector(idata, isize);
        }

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _igrad(
                tscalar* idata, tsize isize,
                const tscalar* wdata,
                const tscalar* odata, tsize osize)
        {
                // input gradient
                tensor::make_vector(idata, isize).noalias() =
                        tensor::make_matrix(wdata, osize, isize).transpose() *
                        tensor::make_vector(odata, osize);
        }

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _pgrad(
                tscalar* idata, tsize isize,
                tscalar* gwdata,
                tscalar* gbdata,
                const tscalar* odata, tsize osize)
        {
                // bias & weights gradient
                tensor::make_vector(gbdata, osize).noalias() =
                        tensor::make_vector(odata, osize);

                tensor::make_matrix(gwdata, osize, isize).noalias() =
                        tensor::make_vector(odata, osize) *
                        tensor::make_vector(idata, isize).transpose();
        }

        linear_layer_t::linear_layer_t(const string_t& parameters)
                :       layer_t(parameters, "fully-connected linear layer, parameters: dims=10[1,4096]")
        {
        }

        size_t linear_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.size();
                const size_t odims = math::clamp(text::from_params<size_t>(configuration(), "dims", 10), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(1, odims, idims);
                m_bdata.resize(odims, 1, 1);

                return psize();
        }

        void linear_layer_t::zero_params()
        {
                m_wdata.zero();
                m_bdata.zero();
        }

        void linear_layer_t::random_params(scalar_t min, scalar_t max)
        {
                m_wdata.random(random_t<scalar_t>(min, max));
                m_bdata.random(random_t<scalar_t>(min, max));
        }

        scalar_t* linear_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::save(m_wdata, params);
                params = tensor::save(m_bdata, params);
                return params;
        }

        const scalar_t* linear_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::load(m_wdata, params);
                params = tensor::load(m_bdata, params);
                return params;
        }

        const tensor_t& linear_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);

                _output(m_idata.data(), isize(),
                        m_wdata.data(), m_bdata.data(),
                        m_odata.data(), osize());

                return m_odata;
        }

        const tensor_t& linear_layer_t::igrad(const tensor_t& output)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata.copy_from(output);

                _igrad(m_idata.data(), isize(),
                       m_wdata.data(),
                       m_odata.data(), osize());

                return m_idata;
        }

        void linear_layer_t::pgrad(const tensor_t& output, scalar_t* gradient)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata.copy_from(output);

                _pgrad(m_idata.data(), isize(),
                       gradient, gradient + m_wdata.size(),
                       m_odata.data(), osize());
        }
}

