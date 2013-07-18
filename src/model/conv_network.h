#ifndef NANOCV_MODEL_CONV_NETWORK_H
#define NANOCV_MODEL_CONV_NETWORK_H

#include "model.h"
#include "conv_layer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // convolution (multi-layer) network model.
        //
        // parameters:
        //      network - default = empty_string -> no hidden layer,
        //              - format = [nconvs : crows : ccols : activation]*, where
        //                          nconvs          - number of convolutions
        //                          crows           - convolution size (rows)
        //                          ccols           - convolution size (columns)
        //                          activation      - activation function id
        /////////////////////////////////////////////////////////////////////////////////////////

        class conv_network_t : public model_t
        {
        public:

                // FIXME: remove this!
                using model_t::resize;
                
                // constructor
                conv_network_t(const string_t& params = string_t());
                conv_network_t(const conv_layer_params_t& params);

                // create an object clone
                virtual rmodel_t clone(const string_t& params) const
                {
                        return rmodel_t(new conv_network_t(params));
                }

                // compute the model output & gradient
                virtual vector_t value(const tensor3d_t& input) const;
                virtual vector_t vgrad(const vector_t& ogradient) const;

                // save/load parameters from vector
                virtual bool save_params(vector_t& x) const;
                virtual bool load_params(const vector_t& x);
                virtual void zero_params();
                virtual void random_params();

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize();

        private:

                // display the model structure
                void print() const;

        private:

                // attributes
                conv_layer_params_t     m_params;       // network parameters (hidden layers)
                conv_layers_t           m_layers;       // convolution layers
        };
}

#endif // NANOCV_MODEL_CONV_NETWORK_H
