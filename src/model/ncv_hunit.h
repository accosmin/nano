#ifndef NANOCV_HUNIT_H
#define NANOCV_HUNIT_H

#include "ncv_serializer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // matrix output unit:
        //	linearly combine (using  = input * conv + bias.
        /////////////////////////////////////////////////////////////////////////////////////////

        class hunit_t
        {
        public:

                // constructor
                hunit_t(size_t n_inputs = 0, size_t n_rows = 0, size_t n_cols = 0);

                // resize to process new inputs (returns the number of parameters to optimize)
                size_t resize(size_t n_inputs, size_t n_rows, size_t n_cols);

                // compute the output & gradient
                scalar_t forward(const matrices_t& input) const;
                void backward(const matrices_t& input, scalar_t gradient);

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
                }

        private:

                // attributes
                matrices_t              m_conv;         // convolution matrices
                matrices_t              m_gconv;        //      (& gradient)
                scalar_t                m_bias;         // bias
                scalar_t                m_gbias;        //      (& gradient)
        };

        typedef std::vector<hunit_t>    ounits_t;
}

#endif // NANOCV_HUNIT_H
