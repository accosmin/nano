#pragma once

#include "layer.h"
#include "conv3d_dmaps.h"

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
        struct convolution_layer_t final : public layer_t
        {
                explicit convolution_layer_t(const string_t& parameters = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t&) override;
                virtual void output(tensor3d_const_map_t, tensor1d_const_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_const_map_t, tensor3d_const_map_t) override;
                virtual void gparam(tensor3d_const_map_t, tensor1d_map_t, tensor3d_const_map_t) override;

                virtual tensor3d_dims_t idims() const override { return m_op.params().idims(); }
                virtual tensor3d_dims_t odims() const override { return m_op.params().odims(); }
                virtual tensor_size_t fanin() const override;
                virtual tensor_size_t psize() const override { return m_op.params().psize(); }
                virtual tensor_size_t flops() const override { return m_op.params().flops_output(); }

        private:

                auto kdims() const { return m_op.params().kdims(); }
                auto bdims() const { return m_op.params().bdims(); }
                auto ksize() const { return nano::size(kdims()); }

                template <typename tmap>
                auto kdata(tmap param) const { return map_tensor(param.data(), kdims()); }
                template <typename tmap>
                auto bdata(tmap param) const { return map_vector(param.data() + ksize(), bdims()); }

                // attributes
                conv3d_dmaps_t  m_op;           ///< 3D convolution operator
        };
}
