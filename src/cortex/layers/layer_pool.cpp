#include "layer_pool.h"
#include "pooling.hpp"
#include "text/from_params.hpp"
#include "math/clamp.hpp"

namespace nano
{
        pool_layer_t::pool_layer_t(const string_t& parameters)
                :       layer_t(parameters),
                        m_alpha(nano::clamp(nano::from_params<scalar_t>(parameters, "dims", 0.1), -100.0, +100.0))
        {
        }

        tensor_size_t pool_layer_t::resize(const tensor3d_t& tensor)
        {
                const auto idims = tensor.size<0>();
                const auto irows = tensor.size<1>();
                const auto icols = tensor.size<2>();

                const auto odims = idims;
                const auto orows = (irows + 1) / 2;
                const auto ocols = (icols + 1) / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_wdata.resize(idims, irows, icols);
                m_sdata.resize(odims, orows, ocols);
                m_cdata.resize(odims, orows, ocols);

                return 0;
        }

        const tensor3d_t& pool_layer_t::output(const tensor3d_t& input)
        {
                assert(idims() == input.size<0>());
                assert(irows() <= input.size<1>());
                assert(icols() <= input.size<2>());

                m_idata = input;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        pooling::output(
                                m_idata.matrix(o), m_alpha,
                                m_wdata.matrix(o),
                                m_sdata.matrix(o),
                                m_cdata.matrix(o),
                                m_odata.matrix(o));
                }

                return m_odata;
        }

        const tensor3d_t& pool_layer_t::ginput(const tensor3d_t& output)
        {
                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());

                m_odata = output;

                for (tensor_size_t o = 0; o < odims(); ++ o)
                {
                        pooling::ginput(
                                m_idata.matrix(o),
                                m_wdata.matrix(o),
                                m_sdata.matrix(o),
                                m_cdata.matrix(o),
                                m_odata.matrix(o));
                }

                return m_idata;
        }

        void pool_layer_t::gparam(const tensor3d_t& output, scalar_t* gradient)
        {
                NANO_UNUSED1(gradient);
                NANO_UNUSED1_RELEASE(output);

                assert(odims() == output.size<0>());
                assert(orows() == output.size<1>());
                assert(ocols() == output.size<2>());
        }
}


