#pragma once

#include "layer.h"
#include "conv4d.h"

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
                explicit convolution_layer_t(const string_t& params = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t&, const string_t&, tensor3d_dims_t&, tensor1d_dims_t&) override;
                virtual void output(const tensor4d_t&, const tensor1d_t&, tensor4d_t&) override;
                virtual void ginput(tensor4d_t&, const tensor1d_t&, const tensor4d_t&) override;
                virtual void gparam(const tensor4d_t&, tensor1d_t&, const tensor4d_t&) override;

                virtual tensor_size_t fanin() const override;
                virtual const probe_t& probe_output() const override { return m_probe_output; }
                virtual const probe_t& probe_ginput() const override { return m_probe_ginput; }
                virtual const probe_t& probe_gparam() const override { return m_probe_gparam; }

        private:

                auto kdims() const { return m_params.kdims(); }
                auto bdims() const { return m_params.bdims(); }
                auto ksize() const { return nano::size(kdims()); }

                template <typename tmap>
                auto kdata(tmap param) const { return map_tensor(param.data(), kdims()); }
                template <typename tmap>
                auto bdata(tmap param) const { return map_vector(param.data() + ksize(), bdims()); }

                // attributes
                conv3d_params_t m_params;       ///<
                conv3d_dmaps_t  m_sparse_op;    ///< 3D convolution operator specialized for sparse mapping
                conv3d_dense_t  m_dense_op;     ///< 3D convolution operator specialized for dense mapping
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
