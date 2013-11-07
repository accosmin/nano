#ifndef NANOCV_CONV_LAYER_HPP
#define NANOCV_CONV_LAYER_HPP

#include "layer.h"
#include "core/logger.h"
#include "core/text.h"
#include "core/math/cast.hpp"
#include "core/math/convolution.hpp"
#include "core/math/clamp.hpp"
#include "core/serializer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // convolution layer:
        //
        // parameters:
        //      convs=16[1,256]         - number of convolutions
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                size_t tcrows,          // convolution size: #rows
                size_t tccols           // convolution size: #cols
        >
        class conv_layer_t : public layer_t
        {
        public:

                // constructor
                conv_layer_t(const string_t& params = string_t())
                        :       m_params(params)
                {
                }

                NCV_MAKE_CLONABLE(conv_layer_t, layer_t, "convolution layer, parameters: convs=16")

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols) { return _resize(idims, irows, icols); }

                // reset parameters
                virtual void zero_params() { _zero_params(); }
                virtual void random_params(scalar_t min, scalar_t max) { _random_params(min, max); }
                virtual void zero_grad() const { _zero_grad(); }

                // serialize parameters & gradients
                virtual serializer_t& save_params(serializer_t& s) const { return s << m_kdata << m_bdata; }
                virtual serializer_t& save_grad(serializer_t& s) const { return s << m_gkdata << m_gbdata; }
                virtual deserializer_t& load_params(deserializer_t& s) { return s >> m_kdata >> m_bdata; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const { return _forward(input); }
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const { return _backward(gradient); }

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const { return _save(oa); }
                virtual bool load(boost::archive::binary_iarchive& ia) { return _load(ia); }

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

        private:

                scalar_t bias(size_t o) const { return m_bdata(o, 0, 0); }
                scalar_t& gbias(size_t o) const { return m_gbdata(o, 0, 0); }

        private:

                //-------------------------------------------------------------------------------------------------

                static void _backward(const matrix_t& ogdata, const matrix_t& kdata, matrix_t& igdata)
                {
                        const int krows = math::cast<int>(kdata.rows());
                        const int kcols = math::cast<int>(kdata.cols());
                        const int kcols4 = kcols - (kcols % 4);

                        const int orows = math::cast<int>(ogdata.rows());
                        const int ocols = math::cast<int>(ogdata.cols());

        //                TODO: move this part to core/convolution.h -> deconv_add()

                        for (int r = 0; r < orows; r ++)
                        {
                                const scalar_t* pogdata = &ogdata(r, 0);

                                for (int kr = 0; kr < krows; kr ++)
                                {
                                        const scalar_t* pkdata = &kdata(kr, 0);
                                        scalar_t* pigdata = &igdata(r + kr, 0);

                                        for (int c = 0; c < ocols; c ++)
                                        {
        //                                        for (int kc = 0; kc < kcols; kc ++)
        //                                        {
        //                                                pigdata[c + kc] += pogdata[c] * pkdata[kc];
        //                                        }

                                                for (int kc = 0; kc < kcols4; kc += 4)
                                                {
                                                        pigdata[c + kc + 0] += pogdata[c] * pkdata[kc + 0];
                                                        pigdata[c + kc + 1] += pogdata[c] * pkdata[kc + 1];
                                                        pigdata[c + kc + 2] += pogdata[c] * pkdata[kc + 2];
                                                        pigdata[c + kc + 3] += pogdata[c] * pkdata[kc + 3];
                                                }
                                                for (int kc = kcols4; kc < kcols; kc ++)
                                                {
                                                        pigdata[c + kc + 0] += pogdata[c] * pkdata[kc];
                                                }
                                        }
                                }
                        }
                }

                //-------------------------------------------------------------------------------------------------

                size_t _resize(size_t idims, size_t irows, size_t icols)
                {
                        const size_t odims = math::clamp(text::from_params<size_t>(m_params, "convs", 16), 1, 256);
                        const size_t crows = tcrows;
                        const size_t ccols = tccols;

                        if (    /*idims < 1 || irows < 1 || icols < 1 ||
                                convs < 1 || crows < 1 || ccols < 1 ||*/
                                irows < crows || icols < ccols)
                        {
                                const string_t message =
                                        "invalid size (" + text::to_string(idims) + "x" + text::to_string(irows) +
                                         "x" + text::to_string(icols) + ") -> (" + text::to_string(odims) + "x" +
                                         text::to_string(crows) + "x" + text::to_string(ccols) + ")";

                                log_error() << "convolution layer: " << message;
                                throw std::runtime_error("convolution layer: " + message);
                        }

                        const size_t orows = irows - crows + 1;
                        const size_t ocols = icols - ccols + 1;

                        m_idata.resize(idims, irows, icols);
                        m_odata.resize(odims, orows, ocols);

                        m_kdata.resize(odims, crows, ccols);
                        m_gkdata.resize(odims, crows, ccols);

                        m_bdata.resize(odims, 1, 1);
                        m_gbdata.resize(odims, 1, 1);

                        return m_kdata.size() + m_bdata.size();
                }

                //-------------------------------------------------------------------------------------------------

                void _zero_params()
                {
                        m_kdata.zero();
                        m_bdata.zero();
                        zero_grad();
                }

                //-------------------------------------------------------------------------------------------------

                void _random_params(scalar_t min, scalar_t max)
                {
                        m_kdata.random(min, max);
                        m_bdata.random(min, max);
                        zero_grad();
                }

                //-------------------------------------------------------------------------------------------------

                void _zero_grad() const
                {
                        m_gkdata.zero();
                        m_gbdata.zero();
                }

                //-------------------------------------------------------------------------------------------------

                bool _save(boost::archive::binary_oarchive& oa) const
                {
                        oa << m_params;
                        oa << m_kdata;
                        oa << m_bdata;

                        return true;
                }

                //-------------------------------------------------------------------------------------------------

                bool _load(boost::archive::binary_iarchive& ia)
                {
                        ia >> m_params;
                        ia >> m_kdata;
                        ia >> m_bdata;

                        return true;
                }

                //-------------------------------------------------------------------------------------------------

                const tensor3d_t& _forward(const tensor3d_t& input) const
                {
                        assert(n_idims() == input.n_dim1());
                        assert(n_irows() <= input.n_rows());
                        assert(n_icols() <= input.n_cols());

                        // convolution output
                        m_idata = input;
                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                matrix_t& odata = m_odata(o);
                                odata.setConstant(bias(o));

                                for (size_t i = 0; i < n_idims(); i ++)
                                {
                                        const matrix_t& idata = m_idata(i);
                                        const matrix_t& kdata = m_kdata(o);

                                        math::conv_add<tcrows, tccols>(idata, kdata, odata);
                                }
                        }

                        return m_odata;
                }

                //-------------------------------------------------------------------------------------------------

                const tensor3d_t& _backward(const tensor3d_t& gradient) const
                {
                        assert(n_odims() == gradient.n_dim1());
                        assert(n_orows() == gradient.n_rows());
                        assert(n_ocols() == gradient.n_cols());

                        // convolution gradient
                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& gdata = gradient(o);

                                // bias
                                gbias(o) += gdata.sum();

                                // kernel
                                for (size_t i = 0; i < n_idims(); i ++)
                                {
                                        const matrix_t& idata = m_idata(i);
                                        matrix_t& gkdata = m_gkdata(o);

                                        math::conv_add<tcrows, tccols>(idata, gdata, gkdata);
                                }
                        }

                        // input gradient
                        m_idata.zero();
                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& odata = gradient(o);

                                for (size_t i = 0; i < n_idims(); i ++)
                                {
                                        const matrix_t& kdata = m_kdata(o);
                                        matrix_t& idata = m_idata(i);

                                        _backward(odata, kdata, idata);
                                }
                        }

                        return m_idata;
                }

                //-------------------------------------------------------------------------------------------------

        private:

                // attributes
                string_t                m_params;

                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer

                tensor3d_t              m_kdata;        // convolution/kernel matrices
                tensor3d_t              m_bdata;        // biases (output)

                mutable tensor3d_t      m_gkdata;       // cumulated gradients
                mutable tensor3d_t      m_gbdata;
        };
}

#endif // NANOCV_CONV_LAYER_HPP

