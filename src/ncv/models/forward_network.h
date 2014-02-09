#ifndef NANOCV_FORWARD_NETWORK_H
#define NANOCV_FORWARD_NETWORK_H

#include "model.h"
#include "layer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // multi-layer feed-forward network model.
        //
        // parameters:  - default = empty_string -> no hidden layer
        //              - format = [layer_id:layer_parameters[,]]*
        /////////////////////////////////////////////////////////////////////////////////////////

        class forward_network_t : public model_t
        {
        public:

                using model_t::resize;
                
                // constructor
                forward_network_t(const string_t& params = string_t());

                // create an object clone
                virtual robject_t clone() const;
                virtual robject_t clone(const std::string& params) const;

                // describe the object
                virtual std::string description() const
                {
                        return "feed-forward network, parameters: [layer_id[:layer_parameters][;]]*";
                }

                // compute the model output
                virtual vector_t value(const tensor3d_t& input) const;

                // save/load/initialize parameters
                virtual bool load_params(const vector_t& x);
                virtual void zero_params();
                virtual void random_params();

                // access current parameters/gradient
                virtual vector_t params() const;
                virtual vector_t gradient(const vector_t& ograd) const;

                // save model description as image
                virtual bool save_as_images(const string_t& basepath) const;

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize();

        private:

                // display the model structure
                void print(const strings_t& layer_ids) const;

        private:

                // attributes
                string_t                m_params;       // network parameters (hidden layers)
                rlayers_t               m_layers;       // feed-forward layers
        };
}

#endif // NANOCV_FORWARD_NETWORK_H
