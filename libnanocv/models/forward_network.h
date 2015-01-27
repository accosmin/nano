#pragma once

#include "libnanocv/model.h"
#include "libnanocv/layer.h"

namespace ncv
{
        ///
        /// multi-layer feed-forward network model
        ///
        class forward_network_t : public model_t
        {
        public:

                NANOCV_MAKE_CLONABLE(forward_network_t, "parameters: [layer_id[:layer_parameters][;]]*")

                using model_t::resize;
                
                ///
                /// \brief constructor
                ///
                forward_network_t(const string_t& parameters = string_t());

                ///
                /// \brief copy constructor
                ///
                forward_network_t(const forward_network_t& other);

                ///
                /// \brief assignment operator
                ///
                forward_network_t& operator=(forward_network_t other);

                ///
                /// \brief compute the model's output
                ///
                virtual const tensor_t& output(const tensor_t& input) const override;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual vector_t gparam(const vector_t& output) const override;

                ///
                /// \brief compute the model's gradient wrt inputs
                ///
                virtual const tensor_t& ginput(const vector_t& output) const override;

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
                virtual size_t psize() const override;

                ///
                /// \brief manage layers
                ///
                size_t n_layers() const { return m_layers.size(); }

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize(bool verbose);

        private:

                ///
                /// \brief display the model structure
                ///
                void print(const strings_t& layer_ids) const;

        private:

                // attributes
                rlayers_t               m_layers;               ///< feed-forward layers
        };
}

