#pragma once

#include "cortex/layer.h"

namespace nano
{
        ///
        /// \brief pooling layer:
        ///     down-sample by 2 using 3x3 overlapping regions with adaptive (learned) 3x3 weights
        ///
        class pooling_ada3x3_layer_t : public layer_t
        {
        public:

                NANO_MAKE_CLONABLE(pooling_ada3x3_layer_t, "adaptive pooling layer using 3x3 overlapping regions")

                // constructor
                explicit pooling_ada3x3_layer_t(const string_t& parameters = string_t());

                // resize to process new tensors of the given type
                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                // reset parameters
                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                // serialize parameters (to memory)
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
                virtual tensor_size_t psize() const override { return m_wdata.size(); }

        private:

                // attributes
                tensor3d_t      m_idata;        ///< input buffer
                tensor3d_t      m_odata;        ///< output buffer
                tensor3d_t      m_wdata;        ///< weights buffer: odims x 3 x 3
        };
}
