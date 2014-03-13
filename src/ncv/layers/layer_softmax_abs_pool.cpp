#include "layer_softmax_abs_pool.h"

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
                const tsize isize = irows * icols;

                const tsize orows = (irows + 1) / 2;
                const tsize ocols = (icols + 1) / 2;
                const tsize osize = orows * ocols;

                for (tsize i = 0; i < isize; i ++)
                {
                        wdata[i] = std::exp(idata[i]);
                }

                for (tsize o = 0; o < osize; o ++)
                {
                        sdata[o] = 0;
                        tdata[o] = 0;
                }

                for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                {
                        for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                        {
                                const tsize iindex = r * icols + c;
                                const tsize oindex = rr * ocols + cc;

                                const tscalar w = wdata[iindex], iw = 1 / w;

                                sdata[oindex] += (w + iw) * idata[r * icols + c];
                                tdata[oindex] += (w + iw);
                        }
                }

                for (tsize o = 0; o < osize; o ++)
                {
                        odata[o] = sdata[o] / tdata[o];
                }
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
                const tsize ocols = (icols + 1) / 2;

                for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                {
                        for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                        {
                                const tsize iindex = r * icols + c;
                                const tsize oindex = rr * ocols + cc;

                                const tscalar w = wdata[iindex], iw = 1 / w;
                                const tscalar i = idata[iindex];
                                const tscalar s = sdata[oindex];
                                const tscalar t = tdata[oindex];

                                idata[iindex] =
                                gdata[oindex] * (t * ((w + iw) + (w - iw) * i) - s * (w - iw)) / (t * t);
                        }
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        size_t softmax_abs_pool_layer_t::resize(const tensor_t& tensor)
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

        const tensor_t& softmax_abs_pool_layer_t::forward(const tensor_t& input)
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

        const tensor_t& softmax_abs_pool_layer_t::backward(const tensor_t& gradient)
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


