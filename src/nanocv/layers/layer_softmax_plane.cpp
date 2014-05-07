#include "layer_softmax_plane.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(
                tscalar* idata, tsize size, tscalar* odata)
        {
                auto imap = tensor::make_vector(idata, size);
                auto omap = tensor::make_vector(odata, size);                
                
                for (tsize i = 0; i < size; i ++)
                {
                        imap(i) = std::exp(imap(i));
                }

                const tscalar sume = imap.sum();
                omap = imap / sume;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(
                tscalar* idata, tsize size, const tscalar* odata, const tscalar* gdata)
        {
                auto imap = tensor::make_vector(idata, size);
                auto omap = tensor::make_vector(odata, size);
                auto gmap = tensor::make_vector(gdata, size);                

                for (tsize i = 0; i < size; i ++)
                {
                        imap(i) = gmap(i) * omap(i) * (1 - omap(i));
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t softmax_plane_layer_t::resize(const tensor_t& tensor)
        {
                const size_t dims = tensor.dims();
                const size_t rows = tensor.rows();
                const size_t cols = tensor.cols();

                m_idata.resize(dims, rows, cols);
                m_odata.resize(dims, rows, cols);

                return 0;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& softmax_plane_layer_t::forward(const tensor_t& input)
        {
                assert(dims() == input.dims());
                assert(rows() == input.rows());
                assert(cols() == input.cols());

                m_idata.copy_from(input);

                for (size_t o = 0; o < dims(); o ++)
                {
                        _forward(m_idata.plane_data(o), m_idata.plane_size(),
                                 m_odata.plane_data(o));
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& softmax_plane_layer_t::backward(const tensor_t& gradient)
        {
                assert(dims() == gradient.dims());
                assert(rows() == gradient.rows());
                assert(cols() == gradient.cols());

                for (size_t o = 0; o < dims(); o ++)
                {
                        _backward(m_idata.plane_data(o), m_idata.plane_size(),
                                  m_odata.plane_data(o),
                                  gradient.plane_data(o));
                }

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


