#ifndef NANOCV_LAYER_LINEAR_H
#define NANOCV_LAYER_LINEAR_H

#include "layer.h"

namespace ncv
{
        ///
        /// \brief fully-connected linear layer (as in MLP models)
        ///
        class linear_layer_t : public layer_t
        {
        public:

                // constructor
                linear_layer_t(const string_t& parameters = string_t());

                // create an object clone
                virtual rlayer_t clone(const string_t& parameters) const
                {
                        return rlayer_t(new linear_layer_t(parameters));
                }

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params();
                virtual void random_params(scalar_t min, scalar_t max);

                // serialize parameters & gradients
                virtual ovectorizer_t& save_params(ovectorizer_t& s) const;
                virtual ovectorizer_t& save_grad(ovectorizer_t& s) const;
                virtual ivectorizer_t& load_params(ivectorizer_t& s);

                // process inputs (compute outputs & gradients)
                virtual const tensor_t& forward(const tensor_t& input);
                virtual const tensor_t& backward(const tensor_t& gradient);

                // compute the number of MFLOPs for the forward/backward pass
                virtual scalar_t forward_mflops() const;
                virtual scalar_t backward_mflops() const;

                // access functions
                virtual const tensor_t& input() const { return m_idata; }
                virtual const tensor_t& output() const { return m_odata; }
                virtual size_t n_parameters() const { return m_wdata.size() + m_bdata.size(); }

        private:

                /////////////////////////////////////////////////////////////////////////////////////////

                size_t isize() const { return m_idata.size(); }
                size_t osize() const { return m_odata.size(); }

                /////////////////////////////////////////////////////////////////////////////////////////

        private:

                // attributes
                tensor_t                m_idata;        ///< input buffer:      isize x 1 x 1
                tensor_t                m_odata;        ///< output buffer:     osize x 1 x 1

                tensor_t                m_wdata;        ///< weights:           1 x osize x isize
                tensor_t                m_bdata;        ///< bias:              osize x 1 x 1

                tensor_t                m_gwdata;       ///< cumulated weight gradients
                tensor_t                m_gbdata;       ///< cumulated bias gradients
        };
}

#endif // NANOCV_LAYER_LINEAR_H
