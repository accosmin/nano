#pragma once

#include "layer.h"
#include "conv4d.h"

namespace nano
{
        ///
        /// \brief fully-connected convolution layer as in convolution networks.
        ///
        /// parameters:
        ///     omaps   - number of output feature planes
        ///     krows   - convolution size
        ///     kcols   - convolution size
        ///     kconn   - connectivity factor: default = 1 (fully connected)
        ///     kdrow   - stride factor for the vertical axis: default = 1
        ///     kdcol   - stride factor for the horizontal axis: default = 1
        ///
        struct convolution_layer_t final : public layer_t
        {
                explicit convolution_layer_t(const string_t& params = string_t());

                virtual rlayer_t clone() const override;
                virtual void configure(const tensor3d_dims_t& idims, const string_t& name) override;
                virtual void output(const tensor4d_t& idata, const tensor1d_t& pdata, tensor4d_t& odata) override;
                virtual void ginput(tensor4d_t& idata, const tensor1d_t& pdata, const tensor4d_t& odata) override;
                virtual void gparam(const tensor4d_t& idata, tensor1d_t& pdata, const tensor4d_t& odata) override;

                virtual tensor_size_t fanin() const override;
                virtual tensor3d_dims_t idims() const override { return m_kernel.params().idims(); }
                virtual tensor3d_dims_t odims() const override { return m_kernel.params().odims(); }
                virtual tensor1d_dims_t pdims() const override { return m_kernel.params().pdims(); }

                virtual const probe_t& probe_output() const override { return m_probe_output; }
                virtual const probe_t& probe_ginput() const override { return m_probe_ginput; }
                virtual const probe_t& probe_gparam() const override { return m_probe_gparam; }

        private:

                auto kdims() const { return m_kernel.params().kdims(); }
                auto ksize() const { return nano::size(kdims()); }

                auto bsize() const { return m_kernel.params().bdims(); }

                auto kdata(tensor1d_t& pdata) const { return map_tensor(pdata.data(), kdims()); }
                auto bdata(tensor1d_t& pdata) const { return map_vector(pdata.data() + ksize(), bsize()); }

                auto kdata(const tensor1d_t& pdata) const { return map_tensor(pdata.data(), kdims()); }
                auto bdata(const tensor1d_t& pdata) const { return map_vector(pdata.data() + ksize(), bsize()); }

                // attributes
                conv4d_t        m_kernel;
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
