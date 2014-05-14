#include "layer_linear.h"
#include "common/math.hpp"
#include "common/random.hpp"
#include "tensor/util.hpp"

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
                tensor::make_vector(odata, osize) =
                        tensor::make_vector(bdata, osize) +
                        tensor::make_matrix(wdata, osize, isize) *
                        tensor::make_vector(idata, isize);
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
                tensor::make_vector(gbdata, osize) =
                        tensor::make_vector(odata, osize);

//                tensor::make_matrix(gwdata, osize, isize) =
//                        tensor::make_vector(odata, osize) *
//                        tensor::make_vector(idata, isize).transpose();

                tsize isize4 = isize & (~tsize(3));
                for (tsize o = 0, k = 0; o < osize; o ++)
                {
                        const tscalar oval = odata[o];
                        for (tsize i = 0; i < isize4; i += 4, k += 4)
                        {
                                gwdata[k + 0] = oval * idata[i + 0];
                                gwdata[k + 1] = oval * idata[i + 1];
                                gwdata[k + 2] = oval * idata[i + 2];
                                gwdata[k + 3] = oval * idata[i + 3];
                        }
                        for (tsize i = isize4; i < isize; i ++, k ++)
                        {
                                gwdata[k + 0] = oval * idata[i];
                        }
                }

                // input gradient
                tensor::make_vector(idata, isize) =
                        tensor::make_matrix(wdata, osize, isize).transpose() *
                        tensor::make_vector(odata, osize);
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
                const size_t odims = math::clamp(text::from_params<size_t>(configuration(), "dims", 10), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(1, odims, idims);
                m_bdata.resize(odims, 1, 1);

                return psize();
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

        scalar_t* linear_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::save(m_wdata, params);
                params = tensor::save(m_bdata, params);
                return params;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const scalar_t* linear_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::load(m_wdata, params);
                params = tensor::load(m_bdata, params);
                return params;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata.copy_from(input);

                _forward(m_idata.data(), isize(),
                         m_wdata.data(), m_bdata.data(),
                         m_odata.data(), osize());

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& linear_layer_t::backward(const tensor_t& output, scalar_t* gradient)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata.copy_from(output);

                _backward(m_idata.data(), isize(),
                          m_wdata.data(), gradient, gradient + m_wdata.size(),
                          m_odata.data(), osize());

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}

