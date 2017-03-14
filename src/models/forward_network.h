#pragma once

#include "model.h"
#include "layer.h"

namespace nano
{
        ///
        /// multi-layer feed-forward network model
        ///
        class forward_network_t : public model_t
        {
        public:

                using model_t::resize;

                ///
                /// \brief constructor
                ///
                explicit forward_network_t(const string_t& parameters = string_t());

                ///
                /// \brief enable copying
                ///
                forward_network_t(const forward_network_t&) = default;

                ///
                /// \brief enable moving
                ///
                forward_network_t(forward_network_t&& other) = default;
                forward_network_t& operator=(forward_network_t&& other) = default;

                ///
                /// \brief clone
                ///
                virtual rmodel_t clone() const override;

                ///
                /// \brief compute the model's output
                ///
                virtual const tensor3d_t& output(const tensor3d_t& input) override;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual const vector_t& gparam(const vector_t& output) override;
                const vector_t& gparam(const tensor3d_t& output);

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor3d_t& ginput(const vector_t& output) override;
                const tensor3d_t& ginput(const tensor3d_t& output);

                ///
                /// \brief retrieve timing information regarding various components
                ///
                virtual timings_t timings() const override;

                ///
                /// \brief
                ///
                virtual void describe() const override;

                ///
                /// \brief save/load/initialize parameters
                ///
                virtual bool load(const vector_t& x) override;
                virtual bool save(vector_t& x) const override;
                virtual void random() override;

                ///
                /// \brief number of parameters
                ///
                virtual tensor_size_t psize() const override;

                ///
                /// \brief number of layers
                ///
                size_t n_layers() const { return m_layers.size(); }

        protected:

                virtual bool resize() override;
                virtual bool save(obstream_t&) const override;
                virtual bool load(ibstream_t&) override;

        private:

                ///
                /// \brief
                ///
                struct layer_info_t
                {
                        layer_info_t(const string_t& name = string_t(), rlayer_t layer = rlayer_t());
                        layer_info_t(const layer_info_t& other);
                        layer_info_t(layer_info_t&&) = default;
                        layer_info_t& operator=(layer_info_t&&) = default;
                        layer_info_t& operator=(const layer_info_t&) = delete;

                        void output(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata);
                        void ginput(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata);
                        void gparam(tensor3d_map_t idata, tensor1d_map_t param, tensor3d_map_t odata);

                        auto idims() const { return m_layer->idims(); }
                        auto odims() const { return m_layer->odims(); }
                        auto isize() const { return nano::size(idims()); }
                        auto osize() const { return nano::size(odims()); }
                        auto xsize() const { return isize() + osize(); }
                        auto psize() const { return m_layer->psize(); }
                        auto flops() const { return m_layer->flops(); }

                        string_t        m_name;
                        rlayer_t        m_layer;
                        timing_t        m_output_timings;
                        timing_t        m_ginput_timings;
                        timing_t        m_gparam_timings;
                };

                using layer_infos_t = std::vector<layer_info_t>;

                // attributes
                layer_infos_t   m_layers;               ///< feed-forward layers
                tensor3d_t      m_idata;                ///< input tensors
                tensor3d_t      m_odata;                ///< output tensors
                vector_t        m_xdata;                ///< buffer: input-output tensors for all layers
                vector_t        m_pdata;                ///< buffer: parameters for all layers
                vector_t        m_gdata;                ///< buffer: parameter gradients for all layers
        };
}
