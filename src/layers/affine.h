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
                virtual void output(const tensor3d_map_t&, const tensor1d_map_t&, const tensor3d_map_t&) override;
                virtual void ginput(const tensor3d_map_t&, const tensor1d_map_t&, const tensor3d_map_t&) override;
                virtual void gparam(const tensor3d_map_t&, const tensor1d_map_t&, const tensor3d_map_t&) override;

                virtual dim3d_t idims() const override { return m_idims; }
                virtual dim3d_t odims() const override { return m_odims; }
                virtual tensor_size_t psize() const override { return m_psize; }
                virtual tensor_size_t flops() const override { return 2 * psize(); }

        private:

                tensor_size_t isize() const { return nano::size(m_idims); }
                tensor_size_t osize() const { return nano::size(m_odims); }

                auto wdata(const tensor1d_map_t& param) const { return map_matrix(param.data(), osize(), isize()); }
                auto bdata(const tensor1d_map_t& param) const { return map_vector(param.data() + osize() * isize(), osize()); }

                // attributes
                dim3d_t         m_idims;
                dim3d_t         m_odims;
                tensor_size_t   m_psize;
        };
}
