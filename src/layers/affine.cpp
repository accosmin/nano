#include "affine.h"
#include "math/random.h"
#include "math/numeric.h"
#include "tensor/numeric.h"
#include "text/to_params.h"
#include "text/from_params.h"
#include "tensor/serialize.h"

namespace nano
{
        affine_layer_t::affine_layer_t(const string_t& parameters) :
                layer_t(to_params(parameters, "dims", "10[1,4096]"))
        {
        }

        rlayer_t affine_layer_t::clone() const
        {
                return std::make_unique<affine_layer_t>(*this);
        }

        tensor_size_t affine_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size();
                const auto odims = nano::clamp(nano::from_params<tensor_size_t>(config(), "dims"), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims());
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(odims, idims);
                m_bdata.resize(odims);

                return psize();
        }

        void affine_layer_t::zero_params()
        {
                tensor::set_zero(m_wdata, m_bdata);
        }

        void affine_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(nano::random_t<scalar_t>(min, max), m_wdata, m_bdata);
        }

        scalar_t* affine_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_wdata, m_bdata);
        }

        const scalar_t* affine_layer_t::load_params(const scalar_t* params)
        {
                return tensor::from_array(params, m_wdata, m_bdata);
        }

        const tensor3d_t& affine_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() == input.size<1>());
                assert(icols() == input.size<2>());

                m_idata = input;

                m_odata.vector() = m_wdata * m_idata.vector() + m_bdata;

                return m_odata;
        }

        const tensor3d_t& affine_layer_t::ginput(const tensor3d_t& output)
        {
                assert(output.size<0>() == odims());
                assert(output.size<1>() == orows());
                assert(output.size<2>() == ocols());

                m_odata = output;

                m_idata.vector() = m_wdata.transpose() * m_odata.vector();

                return m_idata;
        }

        void affine_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(output.size<0>() == odims());
                assert(output.size<1>() == orows());
                assert(output.size<2>() == ocols());

                m_odata = output;

                auto gwdata = tensor::map_matrix(gradient, m_wdata.rows(), m_wdata.cols());
                auto gbdata = tensor::map_vector(gradient + m_wdata.size(), m_bdata.rows());

                gbdata = m_odata.vector();
                gwdata.noalias() = m_odata.vector() * m_idata.vector().transpose();
        }
}

