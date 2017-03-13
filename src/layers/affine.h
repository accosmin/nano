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

                virtual rlayer_t clone() const override;

                virtual tensor_size_t resize(const tensor3d_t& tensor) override;

                virtual void random(const scalar_t min, const scalar_t max) override;
                virtual scalar_t* save_params(scalar_t* params) const override;
                virtual const scalar_t* load_params(const scalar_t* params) override;

                virtual bool save(obstream_t&) const override;
                virtual bool load(ibstream_t&) override;

                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const tensor3d_t& ginput(const tensor3d_t& output) override;
                virtual void gparam(const tensor3d_t& output, scalar_t* gradient) override;

                virtual dim3d_t idims() const override { return m_idata.dims(); }
                virtual dim3d_t odims() const override { return m_odata.dims(); }
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
