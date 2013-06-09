#ifndef NANOCV_OLAYER_H
#define NANOCV_OLAYER_H

#include "ncv_ounit.h"

namespace ncv
{
        class loss_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // vector output layer.
        /////////////////////////////////////////////////////////////////////////////////////////

        class olayer_t
        {
        public:

                // constructor
                olayer_t(size_t n_outputs = 0, size_t n_inputs = 0, size_t n_rows = 0, size_t n_cols = 0);

                // resize to process new inputs (returns the number of parameters to optimize)
                size_t resize(size_t n_outputs, size_t n_inputs, size_t n_rows, size_t n_cols);

                // compute the output & gradient
                vector_t forward(const matrices_t& input) const;
                void forward(const matrices_t& input, const vector_t& target, const loss_t& loss);
                void backward(const matrices_t& input, const vector_t& target, const loss_t& loss);

                // reset parameters
                void zero();
                void random(scalar_t min, scalar_t max);

                // serialize/deserialize parameters
                void serialize(size_t& pos, vector_t& params) const;
                void deserialize(size_t& pos, const vector_t& params);

                // cumulate gradients
                void operator+=(const olayer_t& other);

                // access functions
                scalar_t loss() const { return m_loss; }
                size_t count() const { return m_count; }

        private:

                friend class boost::serialization::access;
                template
                <
                        class tarchive
                >
                void serialize(tarchive & ar, const unsigned int version)
                {
                        ar & m_ounits;
                }

        private:

                // attributes
                ounits_t                m_ounits;
                scalar_t                m_loss;         // cumulated loss value
                size_t                  m_count;        // processed number of samples
        };

        typedef std::vector<olayer_t>   olayers_t;
}

#endif // NANOCV_OLAYER_H
