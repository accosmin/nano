#pragma once

#include "layer.h"
#include "model.h"

namespace nano
{
        ///
        /// \brief multi-layer feed-forward network model.
        ///
        struct forward_network_t final : public model_t
        {
                using model_t::configure;

                ///
                /// \brief constructor
                ///
                forward_network_t(const string_t& config = string_t());

                ///
                /// \brief enable copying
                ///
                forward_network_t(const forward_network_t&);
                forward_network_t& operator=(const forward_network_t&) = delete;

                ///
                /// \brief enable moving
                ///
                forward_network_t(forward_network_t&& other) = default;
                forward_network_t& operator=(forward_network_t&& other) = default;

                virtual rmodel_t clone() const override;
                virtual bool configure(const tensor3d_dims_t& idims, const tensor3d_dims_t& odims) override;

                virtual void random() override;
                virtual const tensor4d_t& output(const tensor4d_t& idata) override;
                virtual const tensor1d_t& gparam(const tensor4d_t& odata) override;
                virtual const tensor4d_t& ginput(const tensor4d_t& odata) override;

                virtual void describe() const override;
                virtual probes_t probes() const override;

                virtual bool save(const string_t& path) const override;
                virtual bool load(const string_t& path) override;

                virtual vector_t params() const override;
                virtual void params(const vector_t&) override;

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
                rlayers_t       m_layers;       ///< feed-forward layers
                tensor1d_t      m_gdata;        ///< gradient wrt parameters
                probe_t         m_probe_output;
                probe_t         m_probe_ginput;
                probe_t         m_probe_gparam;
        };
}
