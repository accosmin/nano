#ifndef NANOCV_SOFTMAX_LAYER_H
#define NANOCV_SOFTMAX_LAYER_H

#include "layer.h"

namespace ncv
{
        ///
        /// soft-max (0, 1) normalization layer.
        ///
        /// parameters:
        ///     type=plane[,global]        - normalization method: by plane or globally
        ///
        class softmax_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(softmax_layer_t)

                // constructor
                softmax_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t, scalar_t) {}

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) { return params; }

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& forward(const tensor_t& input);
                virtual const tensor_t& backward(const tensor_t& output, scalar_t* gradient);

                // access functions
                virtual size_t idims() const { return m_data.dims(); }
                virtual size_t irows() const { return m_data.rows(); }
                virtual size_t icols() const { return m_data.cols(); }
                virtual size_t odims() const { return m_data.dims(); }
                virtual size_t orows() const { return m_data.rows(); }
                virtual size_t ocols() const { return m_data.cols(); }
                virtual size_t psize() const { return 0; }

        private:

                enum class type : int
                {
                        plane,
                        global
                };

        private:

                // attributes
                tensor_t                m_data;         ///< input-output buffer
                type                    m_type;
        };
}

#endif // NANOCV_SOFTMAX_LAYER_H

