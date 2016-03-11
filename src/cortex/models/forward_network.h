#pragma once

#include "cortex/model.h"
#include "cortex/layer.h"

namespace nano
{
        ///
        /// multi-layer feed-forward network model
        ///
        class forward_network_t : public model_t
        {
        public:

                NANO_MAKE_CLONABLE(forward_network_t, "parameters: [layer_id[:layer_parameters][;]]*")

                using model_t::resize;

                ///
                /// \brief constructor
                ///
                explicit forward_network_t(const string_t& parameters = string_t());

                ///
                /// \brief copy constructor
                ///
                forward_network_t(const forward_network_t& other);

                ///
                /// \brief assignment operator
                ///
                forward_network_t& operator=(const forward_network_t&) = delete;

                ///
                /// \brief compute the model's output
                ///
                virtual const tensor_t& output(const tensor_t& input) override;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual const vector_t& gparam(const vector_t& output) override;

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor_t& ginput(const vector_t& output) override;

                ///
                /// \brief save/load/initialize parameters
                ///
                virtual bool load_params(const vector_t& x) override;
                virtual bool save_params(vector_t& x) const override;
                virtual void zero_params() override;
                virtual void random_params() override;

                ///
                /// \brief number of parameters
                ///
                virtual tensor_size_t psize() const override;

                ///
                /// \brief manage layers
                ///
                size_t n_layers() const { return m_layers.size(); }

        protected:

                // resize to new inputs/outputs, returns the number of parameters
                virtual tensor_size_t resize(bool verbose) override;

        private:

                ///
                /// \brief display the model structure
                ///
                void print(const strings_t& layer_ids) const;

        private:

                // attributes
                rlayers_t               m_layers;               ///< feed-forward layers
                vector_t                m_gparam;               ///< buffer gradient wrt parameters
                tensor_t                m_odata;                ///< bufer output
        };
}

