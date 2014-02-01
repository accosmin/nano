#ifndef NANOCV_SOFTMAX_ABS_POOL_LAYER_H
#define NANOCV_SOFTMAX_ABS_POOL_LAYER_H

#include "layer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // softmax absolute pooling layer:
        //      down-sample by 2 from a 3x3 neighbouring region using a soft-max weighting.
        //      weight ~ absolute input value.
        /////////////////////////////////////////////////////////////////////////////////////////

        class softmax_abs_pool_layer_t : public layer_t
        {
        public:

                // constructor
                softmax_abs_pool_layer_t(const string_t& params = string_t());

                NCV_MAKE_CLONABLE(softmax_abs_pool_layer_t, layer_t,
                                  "soft-max absolute pooling layer")

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols);

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const { return s; }
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const { return s; }
                virtual ivectorizer_t& load_params(ivectorizer_t& s) { return s; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const;
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const;

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const { return true; }
                virtual bool load(boost::archive::binary_iarchive& ia) { return true; }

                // save layer description as image
                virtual bool save_as_image(const string_t& basepath) const { return true; }

                // access functions
                virtual size_t n_idims() const { return m_idata.n_dim1(); }
                virtual size_t n_irows() const { return m_idata.n_rows(); }
                virtual size_t n_icols() const { return m_idata.n_cols(); }

                virtual size_t n_odims() const { return m_odata.n_dim1(); }
                virtual size_t n_orows() const { return m_odata.n_rows(); }
                virtual size_t n_ocols() const { return m_odata.n_cols(); }

        private:

                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void forward(
                        const tmatrix& idata, tmatrix& wdata, tmatrix& sdata, tmatrix& tdata,
                        tmatrix& odata) const
                {
                        wdata = idata.array().exp().matrix();

                        sdata.setZero();
                        tdata.setZero();

                        for (auto r = 0, rr = 0; r < idata.rows(); r ++, rr = r / 2)
                        {
                                for (auto c = 0, cc = 0; c < idata.cols(); c ++, cc = c / 2)
                                {
                                        const auto w = wdata(r, c), iw = 1.0 / w;

                                        sdata(rr, cc) += (w + iw) * idata(r, c);
                                        tdata(rr, cc) += (w + iw);
                                }
                        }

                        odata = (sdata.array() / tdata.array()).matrix();
                }

                template
                <
                        typename tmatrix,
                        typename tscalar = typename tmatrix::Scalar
                >
                void backward(
                        const tmatrix& gdata, const tmatrix& wdata, const tmatrix& sdata, const tmatrix& tdata,
                        tmatrix& idata) const
                {
                        for (auto r = 0, rr = 0; r < idata.rows(); r ++, rr = r / 2)
                        {
                                for (auto c = 0, cc = 0; c < idata.cols(); c ++, cc = c / 2)
                                {
                                        const auto w = wdata(r, c), iw = 1.0 / w;
                                        const auto s = sdata(rr, cc);
                                        const auto t = tdata(rr, cc);

                                        idata(r, c) =   gdata(rr, cc) *
                                                        (t * ((w + iw) + (w - iw) * idata(r, c)) - s * (w - iw)) / (t * t);
                                }
                        }
                }

                /////////////////////////////////////////////////////////////////////////////////////////

        private:

                // attributes
                mutable tensor3d_t      m_idata;        // input buffer
                mutable tensor3d_t      m_odata;        // output buffer

                mutable tensor3d_t      m_wdata;        // pooling weights
                mutable tensor3d_t      m_sdata;        // nominator
                mutable tensor3d_t      m_tdata;        // denominator
        };
}

#endif // NANOCV_SOFTMAX_ABS_POOL_LAYER_H

