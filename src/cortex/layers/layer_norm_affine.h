#pragma once

#include "cortex/layer.h"

namespace nano
{
        ///
        /// \brief fully-connected normalized affine layer that works with 1D tensors:
        ///     see "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks",
        ///     by Tim Salimans & Diederik P. Kingma
        ///
        /// parameters:
        ///     dims    - number of output dimensions
        ///
        class norm_affine_layer_t : public layer_t
        {
        public:

                NANO_MAKE_CLONABLE(norm_affine_layer_t, "fully-connected normalized 1D affine layer: dims=10[1,4096]")

                // constructor
                explicit norm_affine_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                // reset parameters
                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                // serialize parameters
                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                // process inputs (compute outputs & gradients)
                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override;
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) override;

                // access functions
                virtual tensor_size_t idims() const override { return m_idata.size<0>(); }
                virtual tensor_size_t irows() const override { return m_idata.size<1>(); }
                virtual tensor_size_t icols() const override { return m_idata.size<2>(); }
                virtual tensor_size_t odims() const override { return m_odata.size<0>(); }
                virtual tensor_size_t orows() const override { return m_odata.size<1>(); }
                virtual tensor_size_t ocols() const override { return m_odata.size<2>(); }
                virtual tensor_size_t psize() const override { return m_vdata.size() + m_gdata.size() + m_bdata.size(); }

        private:

                void update();

        private:

                // attributes
                tensor3d_t      m_idata;        ///< input buffer:      idims x 1 x 1
                tensor3d_t      m_odata;        ///< output buffer:     odims x 1 x 1
                matrix_t        m_vdata;        ///< weights:           odims x idims
                vector_t        m_gdata;        ///< weighting factor:  odims
                vector_t        m_bdata;        ///< bias:              odims

                matrix_t        m_wdata;        ///< normalized weights: g * v / ||v||
        };
}
