#ifndef NANOCV_ACTIVATION_LAYER_H
#define NANOCV_ACTIVATION_LAYER_H

#include "layer.h"
#include "common/math.hpp"

namespace ncv
{
        ///
        /// activation layer: pplies a non-linear scalar function to the each input
        ///
        template
        <
                /// activation value o: o = teval_op(x)
                typename teval_op,

                /// & its gradient wrt to input x, given the output o and propagated gradient g: g = tgrad_op(g, o)
                typename tgrad_op

        >
        class activation_layer_t : public layer_t
        {
        public:

                // destructor
                virtual ~activation_layer_t() {}

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor)
                {
                        return _resize(tensor);
                }

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const { return s; }
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const { return s; }
                virtual ivectorizer_t& load_params(ivectorizer_t& s) { return s; }

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& forward(const tensor_t& input) { return _forward(input); }
                virtual const tensor_t& backward(const tensor_t& gradient) { return _backward(gradient); }

                // access functions
                virtual const tensor_t& input() const { return m_data; }
                virtual const tensor_t& output() const { return m_data; }

        private:

                // resize to process new inputs, returns the number of parameters
                size_t _resize(const tensor_t& tensor)
                {
                        m_data.resize(tensor.dims(), tensor.rows(), tensor.cols());

                        return 0;
                }

                // output
                const tensor_t& _forward(const tensor_t& input)
                {
                        assert(m_data.size() == input.size());

                        math::transform(input, m_data, std::bind(teval_op(), _1));

                        return m_data;
                }

                // gradient
                const tensor_t& _backward(const tensor_t& gradient)
                {
                        assert(m_data.size() == gradient.size());

                        math::transform(gradient, m_data, m_data, std::bind(tgrad_op(), _1, _2));

                        return m_data;
                }

        private:

                // attributes
                tensor_t                m_data;         ///< input-output buffer
        };
}

#endif // NANOCV_ACTIVATION_LAYER_H
