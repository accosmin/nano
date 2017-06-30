#pragma once

#include "layer.h"
#include "conv3d_dense.h"
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
                explicit convolution_layer_t(const string_t& params = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t&, const string_t&) override;
                virtual void output(tensor3d_const_map_t, tensor1d_const_map_t, tensor3d_map_t) override;
                virtual void ginput(tensor3d_map_t, tensor1d_const_map_t, tensor3d_const_map_t) override;
                virtual void gparam(tensor3d_const_map_t, tensor1d_map_t, tensor3d_const_map_t) override;

                virtual tensor3d_dims_t idims() const override { return m_params.idims(); }
                virtual tensor3d_dims_t odims() const override { return m_params.odims(); }
                virtual tensor_size_t fanin() const override;
                virtual tensor_size_t psize() const override { return m_params.psize(); }
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
