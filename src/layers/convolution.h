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
                virtual bool configure(const dim3d_t&) override;
                virtual bool configure(const tensor3d_map_t, const tensor3d_map_t, const vector_map_t) override;

                virtual void output() override;
                virtual void ginput() override;
                virtual void gparam() override;

                virtual dim3d_t idims() const override { return m_idata.dims(); }
                virtual dim3d_t odims() const override { return m_odata.dims(); }
                virtual tensor_size_t psize() const override { return m_op.params().psize(); }
                virtual tensor_size_t flops() const override { return m_op.params().flops_output(); }

        private:

                // attributes
                tensor3d_map_t          m_idata;
                tensor3d_map_t          m_odata;
                vector_map_t            m_param;
                conv3d_toeplitz_t       m_op;           ///< 3D convolution operator
        };
}
