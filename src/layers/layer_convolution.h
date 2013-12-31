#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "layer.h"
#include "util/convolution.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // fixed-size convolution layer:
        //
        // parameters:
        //      count=16[1,256]                 - number of convolutions
        //      rows=8[1,32]                    - size of convolutions (rows)
        //      cols=8[1,32]                    - size of convolutions (columns)
        /////////////////////////////////////////////////////////////////////////////////////////

        class conv_layer_t : public layer_t
        {
        public:

                // constructor
                conv_layer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(conv_layer_t, layer_t,
                                  "convolution layer, parameters: count=16[1,256],rows=8[1,32],cols=8[1,32]")

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const;
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const;
                virtual ivectorizer_t& load_params(ivectorizer_t& s);

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const;
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const;

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

        private:

                /////////////////////////////////////////////////////////////////////////////////////////

                scalar_t bias(size_t o) const { return m_bdata(o, 0, 0); }
                scalar_t weight(size_t o, size_t i) const { return m_wdata(o, i, 0); }

                scalar_t& gweight(size_t o, size_t i) const { return m_gwdata(o, i, 0); }
                scalar_t& gbias(size_t o) const { return m_gbdata(o, 0, 0); }

		bool kmod4x() const { return (m_kdata.n_cols() & 3) == 0; }
                bool omod4x() const { return (m_odata.n_cols() & 3) == 0; }

                template
                <
                        typename tdot
                >
                void forward(tdot dotop) const
                {
                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& kdata = m_kdata(o);
                                matrix_t& odata = m_odata(o);

                                odata.setConstant(bias(o));

                                for (size_t i = 0; i < n_idims(); i ++)
                                {
                                        const matrix_t& idata = m_idata(i);
                                        matrix_t& xdata = m_xdata(o, i);

                                        math::conv_dot<false>(idata, kdata, xdata, dotop);
                                        odata.noalias() += weight(o, i) * xdata;
                                }
                        }
                }

                template
                <
                        typename tmad
                >
                static void backward(const matrix_t& ogdata, const matrix_t& kdata, scalar_t weight, matrix_t& igdata, tmad madop)
                {
                        for (auto r = 0; r < ogdata.rows(); r ++)
                        {
                                const scalar_t* pogdata = &ogdata(r, 0);

                                for (auto kr = 0; kr < kdata.rows(); kr ++)
                                {
                                        const scalar_t* pkdata = &kdata(kr, 0);
                                        scalar_t* pigdata = &igdata(r + kr, 0);

                                        for (auto c = 0; c < ogdata.cols(); c ++)
                                        {
                                                const scalar_t w = weight * pogdata[c];

                                                madop(pkdata, w, pigdata + c, kdata.cols());
                                        }
                                }
                        }
                }

                template
                <
                        typename tdot
                >
                void gbackward(const tensor3d_t& gradient, tdot dotop) const
                {
                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& gdata = gradient(o);
                                matrix_t& gkdata = m_gkdata(o);

                                gkdata.setZero();
                                gbias(o) = gdata.sum();

                                for (size_t i = 0; i < n_idims(); i ++)
                                {
                                        const matrix_t& idata = m_idata(i);
                                        const matrix_t& xdata = m_xdata(o, i);

                                        gweight(o, i) = gdata.cwiseProduct(xdata).sum();
                                        math::wconv_dot<true>(idata, gdata, weight(o, i), gkdata, dotop);
                                }
                        }
                }

                template
                <
                        typename tmad
                >
                void ibackward(const tensor3d_t& gradient, tmad madop) const
                {
                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& gdata = gradient(o);
                                const matrix_t& kdata = m_kdata(o);

                                for (size_t i = 0; i < n_idims(); i ++)
                                {
                                        matrix_t& idata = m_idata(i);

                                        backward(gdata, kdata, weight(o, i), idata, madop);
                                }
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////

        private:

                // attributes
                string_t                m_params;

                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer
                mutable tensor4d_t      m_xdata;        // output convolution buffer

                tensor3d_t              m_kdata;        // convolution/kernel matrices (output)
                tensor3d_t              m_wdata;        // weights (output, input)
                tensor3d_t              m_bdata;        // biases (output)

                mutable tensor3d_t      m_gkdata;       // cumulated gradients
                mutable tensor3d_t      m_gwdata;
                mutable tensor3d_t      m_gbdata;
        };
}

#endif // NANOCV_CONV_LAYER_H

