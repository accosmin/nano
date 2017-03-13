#pragma once

#include "layer.h"
#include "conv3d_toeplitz.h"

namespace nano
{
        ///
        /// \brief fully-connected convolution layer as in convolution networks.
        ///
        /// parameters:
        ///     dims    - number of output planes
        ///     rows    - convolution size
        ///     cols    - convolution size
        ///     conn    - connectivity factor: default = 1 (fully connected)
        ///     drow    - stride factor for the vertical axis: default = 1
        ///     dcol    - stride factor for the horizontal axis: default = 1
        ///
        class NANO_PUBLIC convolution_layer_t final : public layer_t
        {
        public:

                explicit convolution_layer_t(const string_t& parameters = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const dim3d_t&) override;
                virtual void output(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;
                virtual void gparam(tensor3d_map_t, tensor1d_map_t, tensor3d_map_t) override;

                virtual dim3d_t idims() const override { return m_op.params().idims(); }
                virtual dim3d_t odims() const override { return m_op.params().odims(); }
                virtual tensor_size_t psize() const override { return m_op.params().psize(); }
                virtual tensor_size_t flops() const override { return m_op.params().flops_output(); }

        private:

                auto kdims() const { return m_op.params().kdims(); }
                auto bdims() const { return m_op.params().bdims(); }
                auto ksize() const { return nano::size(kdims()); }

                auto kdata(tensor1d_map_t param) const { return map_tensor(param.data(), kdims()); }
                auto bdata(tensor1d_map_t param) const { return map_vector(param.data() + ksize(), bdims()); }

                // attributes
                conv3d_toeplitz_t       m_op;           ///< 3D convolution operator
        };
}
