#include "layer_norm_softmax.h"
#include "common/logger.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(const tscalar* idata, tsize size, tscalar* data)
        {
                auto imap = tensor::make_vector(idata, size);
                auto dmap = tensor::make_vector( data, size);

                dmap = imap.array().exp();

                const tscalar sumd = dmap.sum();
                dmap.noalias() = dmap / sumd;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(const tscalar* gdata, tsize size, tscalar* data)
        {
                auto gmap = tensor::make_vector(gdata, size);
                auto dmap = tensor::make_vector( data, size);

                const tscalar gd = gmap.dot(dmap);
                dmap.noalias() = (dmap.array() * (gmap.array() - gd)).matrix();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        /////////////////////////////////////////////////////////////////////////////////////////

        norm_softmax_layer_t::norm_softmax_layer_t(const string_t& parameters)
                :       layer_t(parameters, "soft-max normalize layer, parameters: type=plane[,global]"),
                        m_type(type::plane)
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t norm_softmax_layer_t::resize(const tensor_t& tensor)
        {
                const string_t t = text::from_params<string_t>(configuration(), "type", "global");
                if (t == "plane")
                {
                        m_type = type::plane;
                }
                else if (t == "global")
                {
                        m_type = type::global;
                }
                else
                {
                        const string_t message = "unknown normalization type <" + t + ">!";

                        log_error() << "normalization layer: " << message;
                        throw std::runtime_error(message);
                }

                const size_t dims = tensor.dims();
                const size_t rows = tensor.rows();
                const size_t cols = tensor.cols();

                m_data.resize(dims, rows, cols);

                return 0;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& norm_softmax_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() == input.rows());
                assert(icols() == input.cols());

                switch (m_type)
                {
                case type::plane:
                        for (size_t o = 0; o < odims(); o ++)
                        {
                                _forward(input.plane_data(o), m_data.plane_size(),
                                         m_data.plane_data(o));
                        }
                        break;

                case type::global:
                default:
                        _forward(input.data(), m_data.size(),
                                 m_data.data());
                }

                return m_data;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& norm_softmax_layer_t::backward(const tensor_t& output, scalar_t*)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                switch (m_type)
                {
                case type::plane:
                        for (size_t o = 0; o < odims(); o ++)
                        {
                                _backward(output.plane_data(o), m_data.plane_size(),
                                          m_data.plane_data(o));
                        }
                        break;

                case type::global:
                default:
                        _backward(output.data(), m_data.size(),
                                  m_data.data());
                        break;
                }

                return m_data;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


