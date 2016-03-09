#include "layer_affine.h"
#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/random.hpp"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"

namespace zob
{
        affine_layer_t::affine_layer_t(const string_t& parameters)
                :       layer_t(parameters)
        {
        }

        tensor_size_t affine_layer_t::resize(const tensor_t& tensor)
        {
                const auto idims = tensor.size();
                const auto odims = zob::clamp(zob::from_params<tensor_size_t>(configuration(), "dims", 10), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims(), tensor.rows(), tensor.cols());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(odims, idims);
                m_bdata.resize(odims);

                return psize();
        }

        void affine_layer_t::zero_params()
        {
                m_wdata.setZero();
                m_bdata.setZero();
        }

        void affine_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(m_wdata, zob::random_t<scalar_t>(min, max));
                tensor::set_random(m_bdata, zob::random_t<scalar_t>(min, max));
        }

        scalar_t* affine_layer_t::save_params(scalar_t* params) const
        {
                params = tensor::to_array(m_wdata, params);
                params = tensor::to_array(m_bdata, params);
                return params;
        }

        const scalar_t* affine_layer_t::load_params(const scalar_t* params)
        {
                params = tensor::from_array(m_wdata, params);
                params = tensor::from_array(m_bdata, params);
                return params;
        }

        const tensor_t& affine_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                m_idata = input;

                m_odata.vector() = m_wdata * m_idata.vector() + m_bdata;

                return m_odata;
        }

        const tensor_t& affine_layer_t::ginput(const tensor_t& output)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata = output;

                m_idata.vector() = m_wdata.transpose() * m_odata.vector();

                return m_idata;
        }

        void affine_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(output.dims() == odims());
                assert(output.rows() == orows());
                assert(output.cols() == ocols());

                m_odata = output;

                auto gwdata = tensor::map_matrix(gradient, m_wdata.rows(), m_wdata.cols());
                auto gbdata = tensor::map_vector(gradient + m_wdata.size(), m_bdata.rows());

                gbdata = m_odata.vector();
                gwdata.noalias() = m_odata.vector() * m_idata.vector().transpose();
        }
}

