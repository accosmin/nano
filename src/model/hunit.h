#ifndef NANOCV_HUNIT_H
#define NANOCV_HUNIT_H

#include "core/serializer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // matrix hidden unit:
        //	affinely combine the inputs using convolutions &
        //      apply a non-linear activation function on all outputs.
        /////////////////////////////////////////////////////////////////////////////////////////

        class hunit_t
        {
        public:

                // constructor
                hunit_t(size_t n_convs = 0, size_t n_conv_rows = 0, size_t n_conv_cols = 0);

                // resize to process new inputs (returns the number of parameters to optimize)
                size_t resize(size_t n_convs, size_t n_conv_rows, size_t n_conv_cols);

                // compute the output & gradient
                const matrix_t& forward(const matrices_t& input) const;
                void backward(const matrices_t& input, const matrix_t& gradient);

                // reset parameters
                void zero();
                void random(scalar_t min, scalar_t max);

                // serialize/deserialize parameters
                void serialize(serializer_t& s) const;
                void gserialize(serializer_t& s) const;
                void deserialize(deserializer_t& s);

                // cumulate gradients
                void operator+=(const hunit_t& other);

        private:

                // convolutions size
                static size_t n_conv_rows() { return 8; }
                static size_t n_conv_cols() { return 8; }

        private:

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_conv;
                        ar & m_gconv;
                        ar & m_bias;
                        ar & m_gbias;
                        ar & m_output;
                }

        private:

                // attributes
                size_t                  m_inputs;
                size_t                  m_rows, m_cols;

                matrices_t              m_conv;         // convolution matrices
                matrices_t              m_gconv;        //      (& gradient)
                scalar_t                m_bias;         // bias
                scalar_t                m_gbias;        //      (& gradient)

                matrix_t                m_output;       //
        };

        typedef std::vector<hunit_t>    hunits_t;
}

#endif // NANOCV_HUNIT_H
