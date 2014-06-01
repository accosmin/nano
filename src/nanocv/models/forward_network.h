#ifndef NANOCV_FORWARD_NETWORK_H
#define NANOCV_FORWARD_NETWORK_H

#include "model.h"
#include "layer.h"

namespace ncv
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
                forward_network_t(const string_t& parameters = string_t());

                ///
                /// \brief create an object clone
                ///
                virtual rmodel_t make(const string_t& configuration) const;
                virtual rmodel_t clone() const;

                ///
                /// \brief compute the model's output
                ///
                virtual const tensor_t& forward(const tensor_t& input) const;

                ///
                /// \brief compute the model's gradient wrt parameters
                ///
                virtual vector_t gradient(const vector_t& output) const;

                ///
                /// \brief compute the model's gradient (wrt parameters & inputs)
                ///
                virtual const tensor_t& backward(const vector_t& output) const;

                ///
                /// \brief save/load/initialize parameters
                ///
                virtual bool load_params(const vector_t& x);
                virtual void zero_params();
                virtual void random_params();

                ///
                /// \brief current parameters
                ///
                virtual vector_t params() const;

                ///
                /// \brief number of parameters
                ///
                virtual size_t psize() const;

                ///
                /// \brief manage layers
                ///
                size_t n_layers() const { return m_layers.size(); }

                bool toggable(size_t layer) const;
                bool enabled(size_t layer) const;

                bool enable(size_t layer);
                bool disable(size_t layer);

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

#endif // NANOCV_FORWARD_NETWORK_H
