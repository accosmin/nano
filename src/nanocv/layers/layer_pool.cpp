#include "layer_pool.h"
#include "common/math.hpp"

namespace ncv
{
        template
        <
                typename tscalar,
                typename tsize
        >
        static void _output(
                const tscalar* idata, tsize irows, tsize icols, tscalar alpha,
                tscalar* wdata, tscalar* sdata, tscalar* cdata, tscalar* odata)
        {
                const tsize orows = (irows + 1) / 2;
                const tsize ocols = (icols + 1) / 2;
                const tscalar ialpha = 1 / alpha;

                auto wmap = tensor::make_matrix(wdata, irows, icols);
                auto smap = tensor::make_matrix(sdata, orows, ocols);
                auto cmap = tensor::make_matrix(cdata, orows, ocols);
                auto omap = tensor::make_matrix(odata, orows, ocols);
                auto imap = tensor::make_matrix(idata, irows, icols);

                wmap = (imap.array() * alpha).exp();
                
                smap.setZero();
                cmap.setZero();

                for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                {
                        for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                        {
                                smap(rr, cc) += wmap(r, c);
                                cmap(rr, cc) += 1;
                        }
                }
                
                omap = ialpha * (smap.array() / cmap.array()).log();
        }

        template
        <
                typename tscalar,
                typename tsize
        >
        static void _igrad(
                tscalar* idata, tsize irows, tsize icols,
                const tscalar* wdata, const tscalar* sdata, const tscalar*, const tscalar* gdata)
        {
                const tsize orows = (irows + 1) / 2;
                const tsize ocols = (icols + 1) / 2;

                auto wmap = tensor::make_matrix(wdata, irows, icols);
                auto smap = tensor::make_matrix(sdata, orows, ocols);
                auto gmap = tensor::make_matrix(gdata, orows, ocols);
                auto imap = tensor::make_matrix(idata, irows, icols);

                for (tsize r = 0, rr = 0; r < irows; r ++, rr = r / 2)
                {
                        for (tsize c = 0, cc = 0; c < icols; c ++, cc = c / 2)
                        {
                                imap(r, c) = gmap(rr, cc) * wmap(r, c) / smap(rr, cc);
                        }
                }
        }
        
        pool_layer_t::pool_layer_t(const string_t& parameters)
                :       layer_t(parameters, "pooling layer, parameters: alpha=[-100.0,+100.0]"),
                        m_alpha(math::clamp(text::from_params<scalar_t>(parameters, "dims", 0.1), -100.0, +100.0))
        {
        }

        size_t pool_layer_t::resize(const tensor_t& tensor)
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
                m_cdata.resize(odims, orows, ocols);

                return 0;
        }

        const tensor_t& pool_layer_t::output(const tensor_t& input)
        {
                assert(idims() == input.dims());
                assert(irows() <= input.rows());
                assert(icols() <= input.cols());

                m_idata.copy_from(input);

                for (size_t o = 0; o < odims(); o ++)
                {
                        _output(m_idata.plane_data(o), irows(), icols(), m_alpha,
                                 m_wdata.plane_data(o),
                                 m_sdata.plane_data(o),
				 m_cdata.plane_data(o),
                                 m_odata.plane_data(o));
                }

                return m_odata;
        }

        const tensor_t& pool_layer_t::igrad(const tensor_t& output)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());

                for (size_t o = 0; o < odims(); o ++)
                {
                        _igrad(m_idata.plane_data(o), irows(), icols(),
                                  m_wdata.plane_data(o),
                                  m_sdata.plane_data(o),
				  m_cdata.plane_data(o),
                                  output.plane_data(o));
                }

                return m_idata;
        }

        void pool_layer_t::pgrad(const tensor_t& output, scalar_t* gradient)
        {
                assert(odims() == output.dims());
                assert(orows() == output.rows());
                assert(ocols() == output.cols());
        }
}


