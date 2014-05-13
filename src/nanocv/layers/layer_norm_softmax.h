#ifndef NANOCV_SOFTMAX_LAYER_PLANE_H
#define NANOCV_SOFTMAX_LAYER_PLANE_H

#include "layer.h"

namespace ncv
{
        ///
        /// soft-max normalize layer.
        ///
        /// parameters:
        ///     type=plane[,global]        - normalization method: by plane or global
        ///
        class norm_softmax_layer_t : public layer_t
        {
        public:

                // constructor
                norm_softmax_layer_t(const string_t& parameters = string_t());

                // create an object clone
                virtual rlayer_t clone(const string_t& parameters) const
                {
                        return rlayer_t(new norm_softmax_layer_t(parameters));
                }

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

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

#endif // NANOCV_SOFTMAX_PLANE_LAYER_H

