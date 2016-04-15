#include "math/clamp.hpp"
#include "math/random.hpp"
#include "tensor/numeric.hpp"
#include "layer_norm_affine.h"
#include "text/from_params.hpp"
#include "tensor/serialize.hpp"

namespace nano
{
        norm_affine_layer_t::norm_affine_layer_t(const string_t& parameters) :
                layer_t(parameters)
        {
        }

        tensor_size_t norm_affine_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size();
                const auto odims = nano::clamp(nano::from_params<tensor_size_t>(configuration(), "dims", 10), 1, 4096);

                // resize buffers
                m_idata.resize(tensor.dims());
                m_odata.resize(odims, 1, 1);

                m_vdata.resize(odims, idims);
                m_gdata.resize(odims);
                m_bdata.resize(odims);

                m_wdata.resize(odims, idims);

                return psize();
        }

        void norm_affine_layer_t::zero_params()
        {
                tensor::set_zero(m_vdata, m_gdata, m_bdata);
                update();
        }

        void norm_affine_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(nano::random_t<scalar_t>(min, max), m_vdata, m_gdata, m_bdata);
                update();
        }

        scalar_t* norm_affine_layer_t::save_params(scalar_t* params) const
        {
                return tensor::to_array(params, m_vdata, m_gdata, m_bdata);
        }

        const scalar_t* norm_affine_layer_t::load_params(const scalar_t* params)
        {
                const auto ret = tensor::from_array(params, m_vdata, m_gdata, m_bdata);
                update();
                return ret;
        }

        void norm_affine_layer_t::update()
        {
                for (auto o = 0; o < odims(); ++ o)
                {
                        m_wdata.row(o) = (m_gdata(o) / m_vdata.row(o).norm()) * m_vdata.row(o);
                }
        }

        const tensor3d_t& norm_affine_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() == input.size<1>());
                assert(icols() == input.size<2>());

                m_idata = input;

                m_odata.vector() = m_wdata * m_idata.vector() + m_bdata;

                return m_odata;
        }

        const tensor3d_t& norm_affine_layer_t::ginput(const tensor3d_t& output)
        {
                assert(output.size<0>() == odims());
                assert(output.size<1>() == orows());
                assert(output.size<2>() == ocols());

                m_odata = output;

                m_idata.vector() = m_wdata.transpose() * m_odata.vector();

                return m_idata;
        }

        void norm_affine_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                assert(output.size<0>() == odims());
                assert(output.size<1>() == orows());
                assert(output.size<2>() == ocols());

                m_odata = output;
                const auto& idata = m_idata.vector();
                const auto& odata = m_odata.vector();

                auto gvdata = tensor::map_matrix(gradient, m_vdata.rows(), m_vdata.cols());
                auto ggdata = tensor::map_vector(gradient + m_vdata.size(), m_gdata.rows());
                auto gbdata = tensor::map_vector(gradient + m_vdata.size() + m_gdata.size(), m_bdata.rows());

                for (auto o = 0; o < odims(); ++ o)
                {
                        const auto vdata = m_vdata.row(o);
                        const auto vv = vdata.dot(vdata);
                        const auto vi = vdata.dot(idata);
                        gvdata.row(o) = (odata(o) * m_gdata(o) / (vv * std::sqrt(vv))) * (idata * vv - vdata.transpose() * vi);
                        ggdata(o) = odata(o) * vi / std::sqrt(vv);
                }
                gbdata = m_odata.vector();
        }
}

