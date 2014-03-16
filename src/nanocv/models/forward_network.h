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
                
                // constructor
                forward_network_t(const string_t& parameters = string_t());

                // create an object clone
                virtual rmodel_t clone(const string_t& parameters) const;

                // compute the model's output
                virtual vector_t value(const tensor_t& input) const;

                // compute the model's gradient
                virtual vector_t gradient(const vector_t& ograd) const;

                // save/load/initialize parameters
                virtual bool load_params(const vector_t& x);
                virtual void zero_params();
                virtual void random_params();

                // access current parameters
                virtual vector_t params() const;

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize(bool verbose);

        private:

                // display the model structure
                void print(const strings_t& layer_ids) const;

        private:

                // attributes
                rlayers_t               m_layers;       // feed-forward layers
        };
}

#endif // NANOCV_FORWARD_NETWORK_H
