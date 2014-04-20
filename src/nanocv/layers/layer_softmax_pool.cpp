#include "layer_softmax_pool.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _forward(
                const tscalar* idata, tsize irows, tsize icols,
                tscalar* wdata, tscalar* sdata, tscalar* tdata, tscalar* odata)
        {
                const tsize orows = (irows + 1) / 2;
                const tsize ocols = (icols + 1) / 2;

                auto wmap = tensor::make_matrix(wdata, irows, icols);
                auto smap = tensor::make_matrix(sdata, orows, ocols);
                auto tmap = tensor::make_matrix(tdata, orows, ocols);
                auto omap = tensor::make_matrix(odata, orows, ocols);
                auto imap = tensor::make_matrix(idata, irows, icols);

                wmap = imap.array().exp();

                smap.setZero();
                tmap.setZero();

                for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                {
                        for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                        {
                                const tscalar w = wmap(r, c);

                                smap(rr, cc) += w * imap(r, c);
                                tmap(rr, cc) += w;
                        }
                }

                omap = smap.array() / tmap.array();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _backward(
                tscalar* idata, tsize irows, tsize icols,
                const tscalar* wdata, const tscalar* sdata, const tscalar* tdata, const tscalar* gdata)
        {
                const tsize orows = (irows + 1) / 2;
                const tsize ocols = (icols + 1) / 2;

                auto wmap = tensor::make_matrix(wdata, irows, icols);
                auto smap = tensor::make_matrix(sdata, orows, ocols);
                auto tmap = tensor::make_matrix(tdata, orows, ocols);
                auto gmap = tensor::make_matrix(gdata, orows, ocols);
                auto imap = tensor::make_matrix(idata, irows, icols);

                for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                {
                        for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                        {
                                const tscalar w = wmap(r, c);
                                const tscalar i = imap(r, c);
                                const tscalar s = smap(rr, cc);
                                const tscalar t = tmap(rr, cc);

                                imap(r, c) =
                                gmap(rr, cc) * (t * (w + w * i) - s * w) / (t * t);
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t softmax_pool_layer_t::resize(const tensor_t& tensor)
        {
                const size_t idims = tensor.dims();
                const size_t irows = tensor.rows();
                const size_t icols = tensor.cols();

                const size_t odims = idims;
                const size_t orows = (irows + 1) / 2;
                const size_t ocols = (icols + 1) / 2;

                m_idata.resize(idims, irows, icols);
                m_odata.resize(odims, orows, ocols);

                m_wdata.resize(idims, irows, icols);
                m_sdata.resize(odims, orows, ocols);
                m_tdata.resize(odims, orows, ocols);

                return 0;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& softmax_pool_layer_t::forward(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() <= input.rows());
                assert(icols() <= input.cols());

                m_idata.copy_from(input);

                for (size_t o = 0; o < odims(); o ++)
                {
                        _forward(m_idata.plane_data(o), irows(), icols(),
                                 m_wdata.plane_data(o),
                                 m_sdata.plane_data(o),
                                 m_tdata.plane_data(o),
                                 m_odata.plane_data(o));
                }

                return m_odata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const tensor_t& softmax_pool_layer_t::backward(const tensor_t& gradient)
        {
                assert(odims() == gradient.dims());
                assert(orows() == gradient.rows());
                assert(ocols() == gradient.cols());

                for (size_t o = 0; o < odims(); o ++)
                {
                        _backward(m_idata.plane_data(o), irows(), icols(),
                                  m_wdata.plane_data(o),
                                  m_sdata.plane_data(o),
                                  m_tdata.plane_data(o),
                                  gradient.plane_data(o));
                }

                return m_idata;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}


