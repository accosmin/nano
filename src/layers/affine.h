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
        class affine_layer_t final : public layer_t
        {
        public:

                explicit affine_layer_t(const string_t& parameters = string_t());

                virtual rlayer_t clone(const string_t& configuration) const override;
                virtual rlayer_t clone() const override;

                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                virtual void zero_params() override;
                virtual void random_params(scalar_t min, scalar_t max) override;

                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override;
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) override;

                virtual tensor_size_t idims() const override { return m_idata.size<0>(); }
                virtual tensor_size_t irows() const override { return m_idata.size<1>(); }
                virtual tensor_size_t icols() const override { return m_idata.size<2>(); }
                virtual tensor_size_t odims() const override { return m_odata.size<0>(); }
                virtual tensor_size_t orows() const override { return m_odata.size<1>(); }
                virtual tensor_size_t ocols() const override { return m_odata.size<2>(); }
                virtual tensor_size_t psize() const override { return m_wdata.size() + m_bdata.size(); }
                virtual tensor_size_t flops() const override { return psize(); }

        private:

                // attributes
                tensor3d_t      m_idata;        ///< input buffer:      idims x 1 x 1
                tensor3d_t      m_odata;        ///< output buffer:     odims x 1 x 1
                matrix_t        m_wdata;        ///< weights:           odims x idims
                vector_t        m_bdata;        ///< bias:              odims
        };
}
