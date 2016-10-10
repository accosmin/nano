#pragma once

#include "layer.h"

namespace nano
{
        ///
        /// \brief fully-connected affine layer that works with 1D tensors (as in MLP models).
        ///
        /// parameters:
        ///     dims    - number of output dimensions
        ///
        class affine_layer_t : public layer_t
        {
        public:

                NANO_MAKE_CLONABLE(affine_layer_t)

                // constructor
                explicit affine_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor3d_t& tensor) final;

                // reset parameters
                virtual void zero_params() final;
                virtual void random_params(scalar_t min, scalar_t max) final;

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const final;
                virtual const scalar_t* load_params(const scalar_t* params) final;

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& output(const tensor3d_t& input) final;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) final;
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) final;

                // access functions
                virtual tensor_size_t idims() const final { return m_idata.size<0>(); }
                virtual tensor_size_t irows() const final { return m_idata.size<1>(); }
                virtual tensor_size_t icols() const final { return m_idata.size<2>(); }
                virtual tensor_size_t odims() const final { return m_odata.size<0>(); }
                virtual tensor_size_t orows() const final { return m_odata.size<1>(); }
                virtual tensor_size_t ocols() const final { return m_odata.size<2>(); }
                virtual tensor_size_t psize() const final { return m_wdata.size() + m_bdata.size(); }
                virtual tensor_size_t flops() const final { return psize(); }

        private:

                // attributes
                tensor3d_t      m_idata;        ///< input buffer:      idims x 1 x 1
                tensor3d_t      m_odata;        ///< output buffer:     odims x 1 x 1
                matrix_t        m_wdata;        ///< weights:           odims x idims
                vector_t        m_bdata;        ///< bias:              odims
        };
}
