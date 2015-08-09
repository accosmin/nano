#include "layer_linear.h"
#include "linear.hpp"
#include "nanocv/text.hpp"
#include "nanocv/math/clamp.hpp"
#include "nanocv/math/random.hpp"
#include "nanocv/tensor/random.hpp"
#include "nanocv/tensor/serialize.hpp"

namespace ncv
{
        linear_layer_t::linear_layer_t(const string_t& parameters)
                :       layer_t(parameters)
        {
        }

        size_t linear_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.size();
                const size_t odims = math::clamp(text::from_params<size_t>(configuration(), "dims", 10),
                                                 size_t(1), size_t(4096));

                // resize buffers
                m_idata.resize(idims, 1, 1);
                m_odata.resize(odims, 1, 1);

                m_wdata.resize(odims, idims);
                m_bdata.resize(odims);

                return psize();
        }

        void linear_layer_t::zero_params()
        {
                m_wdata.setZero();
                m_bdata.setZero();
        }

        void linear_layer_t::random_params(scalar_t min, scalar_t max)
        {
                tensor::set_random(m_wdata, random_t<scalar_t>(min, max));
                tensor::set_random(m_bdata, random_t<scalar_t>(min, max));
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
                assert(idims() == static_cast<size_t>(input.dims()));
                assert(irows() == static_cast<size_t>(input.rows()));
                assert(icols() == static_cast<size_t>(input.cols()));

                m_idata = input;

                linear::output(m_idata, m_wdata, m_bdata, m_odata);

                return m_odata;
        }

        const tensor_t& linear_layer_t::ginput(const tensor_t& output)
        {
                assert(static_cast<size_t>(output.dims()) == odims());
                assert(static_cast<size_t>(output.rows()) == orows());
                assert(static_cast<size_t>(output.cols()) == ocols());

                m_odata = output;

                linear::ginput(m_idata, m_wdata, m_bdata, m_odata);

                return m_idata;
        }

        void linear_layer_t::gparam(const tensor_t& output, scalar_t* gradient)
        {
                assert(static_cast<size_t>(output.dims()) == odims());
                assert(static_cast<size_t>(output.rows()) == orows());
                assert(static_cast<size_t>(output.cols()) == ocols());

                m_odata = output;

                linear::gparam(m_idata,
                               tensor::map_matrix(gradient, m_wdata.rows(), m_wdata.cols()),
                               tensor::map_vector(gradient + m_wdata.size(), m_bdata.rows()),
                               m_odata);
        }
}

