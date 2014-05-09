#include "layer_norm_softabs.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(const tscalar* idata, tsize size, tscalar* data, tscalar* wdata)
        {
                auto imap = tensor::make_vector(idata, size);
                auto dmap = tensor::make_vector( data, size);
                auto wmap = tensor::make_vector(wdata, size);

                dmap = imap.array().exp();
                wmap = dmap.array() - 1.0 / dmap.array();
                dmap.noalias() = (dmap.array() + 1.0 / dmap.array()).matrix();

                const tscalar sumd = dmap.sum();
                dmap.noalias() = dmap / sumd;
                wmap.noalias() = wmap / sumd;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(const tscalar* gdata, tsize size, tscalar* data, const tscalar* wdata)
        {
                auto gmap = tensor::make_vector(gdata, size);
                auto dmap = tensor::make_vector( data, size);
                auto wmap = tensor::make_vector(wdata, size);

                const tscalar gd = gmap.dot(dmap);
                dmap.noalias() = (wmap.array() * (gmap.array() - gd)).matrix();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t norm_softabs_layer_t::resize(const tensor_t& tensor)
        {
                const size_t dims = tensor.dims();
                const size_t rows = tensor.rows();
                const size_t cols = tensor.cols();

                m_data.resize(dims, rows, cols);
                m_wdata.resize(dims, rows, cols);

                return 0;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& norm_softabs_layer_t::forward(const tensor_t& input)
        {
                assert(dims() == input.dims());
                assert(rows() == input.rows());
                assert(cols() == input.cols());

                for (size_t o = 0; o < dims(); o ++)
                {
                        _forward(input.plane_data(o), m_data.plane_size(),
                                 m_data.plane_data(o), m_wdata.plane_data(o));
                }

                return m_data;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& norm_softabs_layer_t::backward(const tensor_t& gradient)
        {
                assert(dims() == gradient.dims());
                assert(rows() == gradient.rows());
                assert(cols() == gradient.cols());

                for (size_t o = 0; o < dims(); o ++)
                {
                        _backward(gradient.plane_data(o), m_data.plane_size(),
                                  m_data.plane_data(o), m_wdata.plane_data(o));
                }

                return m_data;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


