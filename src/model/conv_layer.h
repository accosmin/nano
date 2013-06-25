#ifndef NANOCV_CONV_LAYER_H
#define NANOCV_CONV_LAYER_H

#include "core/tensor3d.h"
#include "core/tensor4d.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // convolution layer:
        //      - process a set of inputs of size (irows, icols) and produces
        //              a set of outputs using convolution matrices of size (crows, ccols).
        /////////////////////////////////////////////////////////////////////////////////////////

        class conv_layer_t
        {
        public:

                // constructor
                conv_layer_t(size_t inputs = 0, size_t irows = 0, size_t icols = 0,
                             size_t outputs = 0, size_t crows = 0, size_t ccols = 0);

                // resize to new dimensions
                size_t resize(size_t inputs, size_t irows, size_t icols,
                              size_t outputs, size_t crows, size_t ccols);

                // reset parameters
                void zero();
                void random(scalar_t min = -0.1, scalar_t max = 0.1);
                void zero_grad();

                // process inputs (compute outputs & gradients)
                const tensor3d_t& forward(const tensor3d_t& input) const;
                const tensor3d_t& backward(const tensor3d_t& gradient);

                // serialize/deserialize data
                friend serializer_t& operator<<(serializer_t& s, const conv_layer_t& layer);
                friend deserializer_t& operator>>(deserializer_t& s, conv_layer_t& layer);

                // access functions
                size_t n_inputs() const { return m_idata.n_dim1(); }
                size_t n_irows() const { return m_idata.n_rows(); }
                size_t n_icols() const { return m_idata.n_cols(); }

                size_t n_outputs() const { return m_odata.n_dim1(); }
                size_t n_orows() const { return m_odata.n_rows(); }
                size_t n_ocols() const { return m_odata.n_cols(); }

                const tensor4d_t& gdata() const { return m_gdata; }

        private:

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_idata;
                        ar & m_cdata;
                        ar & m_gdata;
                        ar & m_odata;
                }

        private:

                // attributes
                mutable tensor3d_t      m_idata;        // input buffer
                tensor4d_t              m_cdata;        // convolution matrices
                tensor4d_t              m_gdata;        // cumulated gradient of the convolution matrices
                mutable tensor3d_t      m_odata;        // output buffer
        };

        // serialize/deserialize data
        inline serializer_t& operator<<(serializer_t& s, const conv_layer_t& layer)
        {
                return s << layer.m_cdata;
        }

        inline deserializer_t& operator>>(deserializer_t& s, conv_layer_t& layer)
        {
                return s >> layer.m_cdata;
        }
}

#endif // NANOCV_CONV_LAYER_H
