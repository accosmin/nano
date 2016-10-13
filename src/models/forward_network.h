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

                NANO_MAKE_CLONABLE(forward_network_t)

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
                /// \brief compute the model's output
                ///
                virtual const tensor3d_t& output(const tensor3d_t& input) override final;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual const vector_t& gparam(const vector_t& output) override final;
                const vector_t& gparam(const tensor3d_t& output);

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor3d_t& ginput(const vector_t& output) override final;
                const tensor3d_t& ginput(const tensor3d_t& output);

                ///
                /// \brief retrieve timing information regarding various components
                ///
                virtual timings_t timings() const override final;

                ///
                /// \brief save/load/initialize parameters
                ///
                virtual bool load_params(const vector_t& x) override final;
                virtual bool save_params(vector_t& x) const override final;
                virtual void zero_params() override final;
                virtual void random_params() override final;

                ///
                /// \brief number of parameters
                ///
                virtual tensor_size_t psize() const override final;

                ///
                /// \brief manage layers
                ///
                size_t n_layers() const { return m_layers.size(); }

        protected:

                // resize to new inputs/outputs, returns the number of parameters
                virtual tensor_size_t resize(bool verbose) override final;

        private:

                ///
                /// \brief display the model structure
                ///
                void print() const;

                ///
                /// \brief
                ///
                struct layer_info_t
                {
                        layer_info_t(const string_t& name = string_t(), rlayer_t layer = rlayer_t());
                        layer_info_t(const layer_info_t& other);

                        layer_info_t(layer_info_t&&) = default;
                        layer_info_t& operator=(layer_info_t&&) = default;

                        const tensor3d_t& output(const tensor3d_t&);
                        const tensor3d_t& ginput(const tensor3d_t&);
                        scalar_t* gparam(const tensor3d_t&, scalar_t*);

                        string_t        m_name;
                        rlayer_t        m_layer;
                        timing_t        m_output_timings;
                        timing_t        m_ginput_timings;
                        timing_t        m_gparam_timings;
                };

                using layer_infos_t = std::vector<layer_info_t>;

        private:

                // attributes
                layer_infos_t           m_layers;       ///< feed-forward layers
                vector_t                m_gparam;       ///< buffer gradient wrt parameters
                tensor3d_t              m_odata;        ///< bufer output
        };
}

