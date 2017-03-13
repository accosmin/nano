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
                virtual bool configure(const dim3d_t&) override;
                virtual bool configure(const tensor3d_map_t, const tensor3d_map_t, const vector_map_t) override;

                virtual void output() override;
                virtual void ginput() override;
                virtual void gparam() override;

                virtual dim3d_t idims() const override { return m_idata.dims(); }
                virtual dim3d_t odims() const override { return m_odata.dims(); }
                virtual tensor_size_t psize() const override { return m_param.size(); }
                virtual tensor_size_t flops() const override { return psize(); }

        private:

                // attributes
                tensor3d_map_t          m_idata;
                tensor3d_map_t          m_odata;
                vector_map_t            m_param;
        };
}
