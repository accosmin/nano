#ifndef NANOCV_POOL_SOFTABS_LAYER_H
#define NANOCV_POOL_SOFTABS_LAYER_H

#include "layer.h"

namespace ncv
{
        ///
        /// soft-abs pooling layer:
        ///      down-sample by 2 from a 3x3 neighbouring region using a soft-max weighting.
        ///      weight ~ absolute input value.
        ///
        class pool_softabs_layer_t : public layer_t
        {
        public:

                NANOCV_MAKE_CLONABLE(pool_softabs_layer_t)

                // constructor
                pool_softabs_layer_t(const string_t& parameters = string_t())
                        :       layer_t(parameters, "soft-abs pooling layer")
                {
                }

                // resize to process new tensors of the given type
                virtual size_t resize(const tensor_t& tensor);

                // reset parameters
                virtual void zero_params() {}
                virtual void random_params(scalar_t min, scalar_t max) {}

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const { return params; }
                virtual const scalar_t* load_params(const scalar_t* params) { return params; }

                // process inputs (compute outputs)
                virtual const tensor_t& forward(const tensor_t& input);
                virtual const tensor_t& backward(const tensor_t& output, scalar_t* gradient);

                // access functions
                virtual size_t idims() const { return m_idata.dims(); }
                virtual size_t irows() const { return m_idata.rows(); }
                virtual size_t icols() const { return m_idata.cols(); }
                virtual size_t odims() const { return m_odata.dims(); }
                virtual size_t orows() const { return m_odata.rows(); }
                virtual size_t ocols() const { return m_odata.cols(); }
                virtual size_t psize() const { return 0; }

        private:

                // attributes
                tensor_t                m_idata;        ///< input buffer
                tensor_t                m_odata;        ///< output buffer

                tensor_t                m_wdata;        ///< pooling weights
                tensor_t                m_sdata;        ///< nominator
                tensor_t                m_tdata;        ///< denominator
        };
}

#endif // NANOCV_POOL_SOFTABS_LAYER_H

