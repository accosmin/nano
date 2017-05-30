#pragma once

#include "layer.h"
#include "model.h"

namespace nano
{
        ///
        /// multi-layer feed-forward network model
        ///
        struct forward_network_t final : public model_t
        {
                using model_t::model_t;
                using model_t::configure;

                ///
                /// \brief enable copying
                ///
                forward_network_t(const forward_network_t&);

                ///
                /// \brief enable moving
                ///
                forward_network_t(forward_network_t&& other) = default;
                forward_network_t& operator=(forward_network_t&& other) = default;

                virtual rmodel_t clone() const override;
                virtual bool configure(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims) override;

                virtual void random() override;
                virtual const tensor3d_t& output(const tensor3d_t& input) override;
                virtual const vector_t& gparam(const vector_t& output) override;
                virtual const tensor3d_t& ginput(const vector_t& output) override;

                virtual void describe() const override;
                virtual probes_t probes() const override;

                virtual bool save(const string_t& path) const override;
                virtual bool load(const string_t& path) override;

                virtual const vector_t& params() const override;
                virtual void params(const vector_t& x) override;

                virtual tensor_size_t psize() const override;
                virtual tensor3d_dims_t idims() const override;
                virtual tensor3d_dims_t odims() const override;

                ///
                /// \brief number of layers
                ///
                size_t n_layers() const { return m_layers.size(); }

        private:

                using rlayers_t = std::vector<rlayer_t>;

                // attributes
                tensor3d_dims_t m_idims;        ///<
                tensor3d_dims_t m_odims;        ///<
                rlayers_t       m_layers;       ///< feed-forward layers
                tensor3d_t      m_idata;        ///< input tensor
                tensor3d_t      m_odata;        ///< output tensor
                vector_t        m_xdata;        ///< buffer: concatenated input-output tensors for all layers
                vector_t        m_pdata;        ///< buffer: parameters for all layers
                vector_t        m_gdata;        ///< buffer: parameter gradients for all layers
        };
}
