#ifndef NANOCV_ACTIVATION_LAYER_H
#define NANOCV_ACTIVATION_LAYER_H

#include "layer.h"
#include "core/math/transform.hpp"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // activation layer:
        //      applies a non-linear scalar function to the each input.
        /////////////////////////////////////////////////////////////////////////////////////////

        template
        <
                // activation value o: o = teval_op(x)
                typename teval_op,

                // & its gradient wrt to input x, given the output o and propagated gradient g: g = tgrad_op(g, o)
                typename tgrad_op

        >
        class activation_layer_t : public layer_t
        {
        public:

                // destructor
                virtual ~activation_layer_t() {}

                // resize to process new inputs, returns the number of parameters
                virtual size_t resize(size_t idims, size_t irows, size_t icols)
                {
                        return _resize(idims, irows, icols);
                }

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}
                virtual void zero_grad() const {}

                // serialize parameters & gradients
                virtual serializer_t& save_params(serializer_t& s) const { return s; }
                virtual serializer_t& save_grad(serializer_t& s) const { return s; }
                virtual deserializer_t& load_params(deserializer_t& s) { return s; }

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& forward(const tensor3d_t& input) const { return _forward(input); }
                virtual const tensor3d_t& backward(const tensor3d_t& gradient) const { return _backward(gradient); }

                // save/load parameters to/from file
                virtual bool save(boost::archive::binary_oarchive& oa) const { return true; }
                virtual bool load(boost::archive::binary_iarchive& ia) { return true; }

                // access functions
                virtual size_t n_idims() const { return m_data.n_dim1(); }
                virtual size_t n_irows() const { return m_data.n_rows(); }
                virtual size_t n_icols() const { return m_data.n_cols(); }

                virtual size_t n_odims() const { return m_data.n_dim1(); }
                virtual size_t n_orows() const { return m_data.n_rows(); }
                virtual size_t n_ocols() const { return m_data.n_cols(); }

        private:

                // resize to process new inputs, returns the number of parameters
                size_t _resize(size_t idims, size_t irows, size_t icols)
                {
                        m_data.resize(idims, irows, icols);

                        return 0;
                }

                // output
                const tensor3d_t& _forward(const tensor3d_t& input) const
                {
                        assert(n_idims() == input.n_dim1());
                        assert(n_irows() <= input.n_rows());
                        assert(n_icols() <= input.n_cols());

                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& idata = input(o);
                                matrix_t& odata = m_data(o);

                                math::transform(idata, odata, std::bind(teval_op(), _1));
                        }

                        return m_data;
                }

                // gradient
                const tensor3d_t& _backward(const tensor3d_t& gradient) const
                {
                        assert(n_odims() == gradient.n_dim1());
                        assert(n_orows() == gradient.n_rows());
                        assert(n_ocols() == gradient.n_cols());

                        for (size_t o = 0; o < n_odims(); o ++)
                        {
                                const matrix_t& gdata = gradient(o);
                                const matrix_t& odata = m_data(o);
                                matrix_t& idata = m_data(o);

                                math::transform(gdata, odata, idata, std::bind(tgrad_op(), _1, _2));
                        }

                        return m_data;
                }

        private:

                // attributes
                mutable tensor3d_t      m_data;         // input-output buffer
        };
}

#endif // NANOCV_ACTIVATION_LAYER_H
