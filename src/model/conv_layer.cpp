#include "conv_layer.h"
#include "core/logger.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        static void forward(const matrix_t& idata, const matrix_t& cdata, matrix_t& odata)
        {
                const size_t crows = static_cast<size_t>(cdata.rows());
                const size_t ccols = static_cast<size_t>(cdata.cols());

                const size_t orows = static_cast<size_t>(idata.rows() - crows + 1);
                const size_t ocols = static_cast<size_t>(idata.cols() - ccols + 1);

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                odata(r, c) += idata.block(r, c, crows, ccols).cwiseProduct(cdata).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void gradient(const matrix_t& in, const matrix_t& gd, matrix_t& co_gd)
        {
                const size_t crows = static_cast<size_t>(co_gd.rows());
                const size_t ccols = static_cast<size_t>(co_gd.cols());

                const size_t orows = static_cast<size_t>(in.rows() - crows + 1);
                const size_t ocols = static_cast<size_t>(in.cols() - ccols + 1);

                for (size_t r = 0; r < crows; r ++)
                {
                        for (size_t c = 0; c < ccols; c ++)
                        {
                                co_gd(r, c) += in.block(r, c, orows, ocols).cwiseProduct(gd).sum();
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        static void backward(const matrix_t& gd, const matrix_t& co, matrix_t& in_gd)
        {
                const size_t crows = static_cast<size_t>(co.rows());
                const size_t ccols = static_cast<size_t>(co.cols());

                const size_t orows = static_cast<size_t>(gd.rows());
                const size_t ocols = static_cast<size_t>(gd.cols());

                for (size_t r = 0; r < orows; r ++)
                {
                        for (size_t c = 0; c < ocols; c ++)
                        {
                                // FIXME: can this be written more efficiently as a block operation?!
                                for (size_t rr = 0; rr < crows; rr ++)
                                {
                                        for (size_t cc = 0; cc < ccols; cc ++)
                                        {
                                                in_gd(r + rr, c + cc) += gd(r, c) * co(rr, cc);
                                        }
                                }
                        }
                }
        }

        //-------------------------------------------------------------------------------------------------

        conv_layer_t::conv_layer_t(size_t inputs, size_t irows, size_t icols,
                                   size_t outputs, size_t crows, size_t ccols)
        {
                resize(inputs, irows, icols, outputs, crows, ccols);
        }

        //-------------------------------------------------------------------------------------------------

        size_t conv_layer_t::resize(size_t inputs, size_t irows, size_t icols,
                                    size_t outputs, size_t crows, size_t ccols)
        {
                if (    /*inputs < 1 || irows < 1 || icols < 1 ||
                        outputs < 1 || crows < 1 || ccols < 1 ||*/
                        irows < crows || icols < ccols)
                {
                        log_warning() << "convolution layer: invalid size ("
                                      << inputs << "x" << irows << "x" << icols
                                      << ") -> (" << outputs << "x" << crows << "x" << ccols << ")";
                        return 0;
                }

                m_idata.resize(inputs, irows, icols);
                m_cdata.resize(outputs, inputs, crows, ccols);
                m_gdata.resize(outputs, inputs, crows, ccols);
                m_odata.resize(outputs, irows - crows + 1, icols - ccols + 1);

                return m_cdata.size();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::zero()
        {
                m_cdata.zero();
                m_gdata.zero();
        }

        //-------------------------------------------------------------------------------------------------

        void conv_layer_t::random(scalar_t min, scalar_t max)
        {
                m_cdata.random(min, max);
                m_gdata.zero();
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::forward(const tensor3d_t& input) const
        {
                assert(n_inputs() == input.n_dim1());
                assert(n_irows() <= input.n_rows());
                assert(n_icols() <= input.n_cols());

                m_idata = input;

                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        matrix_t& odata = m_odata(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                const matrix_t& cdata = m_cdata(o, i);

                                ncv::forward(idata, cdata, odata);
                        }
                }

                return m_odata;
        }

        //-------------------------------------------------------------------------------------------------

        const tensor3d_t& conv_layer_t::backward(const tensor3d_t& gradient)
        {
                assert(n_outputs() == gradient.n_dim1());
                assert(n_orows() == gradient.n_rows());
                assert(n_ocols() == gradient.n_cols());

                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        const matrix_t& ogdata = gradient(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& idata = m_idata(i);
                                matrix_t& gdata = m_gdata(o, i);

                                ncv::gradient(idata, ogdata, gdata);
                        }
                }

                for (size_t o = 0; o < n_outputs(); o ++)
                {
                        const matrix_t& ogdata = gradient(o);

                        for (size_t i = 0; i < n_inputs(); i ++)
                        {
                                const matrix_t& cdata = m_cdata(o, i);
                                matrix_t& igdata = m_idata(i);

                                ncv::backward(ogdata, cdata, igdata);
                        }
                }

                return m_idata;
        }

        //-------------------------------------------------------------------------------------------------
}

