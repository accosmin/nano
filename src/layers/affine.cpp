#include "affine.h"
#include "io/ibstream.h"
#include "io/obstream.h"
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

        void affine_layer_t::random_params(scalar_t min, scalar_t max)
        {
                nano::set_random(nano::random_t<scalar_t>(min, max), m_wdata, m_bdata);
        }

        scalar_t* affine_layer_t::save_params(scalar_t* params) const
        {
                return nano::to_array(params, m_wdata, m_bdata);
        }

        const scalar_t* affine_layer_t::load_params(const scalar_t* params)
        {
                return nano::from_array(params, m_wdata, m_bdata);
        }

        bool affine_layer_t::save(obstream_t& ob) const
        {
                return  ob.write_matrix(m_wdata) &&
                        ob.write_vector(m_bdata);
        }

        bool affine_layer_t::load(ibstream_t& ib)
        {
                return  ib.read_matrix(m_wdata) &&
                        ib.read_vector(m_bdata);
        }

        const tensor3d_t& affine_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.dims());

                m_idata = input;

                m_odata.vector() = m_wdata * m_idata.vector() + m_bdata;

                return m_odata;
        }

        const tensor3d_t& affine_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.dims());

                m_odata = output;

                m_idata.vector() = m_wdata.transpose() * m_odata.vector();

                return m_idata;
        }

        void affine_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());

                m_odata = output;

                auto gwdata = nano::map_matrix(gradient, m_wdata.rows(), m_wdata.cols());
                auto gbdata = nano::map_vector(gradient + m_wdata.size(), m_bdata.rows());

                gbdata = m_odata.vector();
                gwdata.noalias() = m_odata.vector() * m_idata.vector().transpose();
        }
}

