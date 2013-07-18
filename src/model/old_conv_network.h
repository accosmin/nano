#ifndef NANOCV_MODEL_CONV_NETWORK_H
#define NANOCV_MODEL_CONV_NETWORK_H

#include "model.h"
#include "conv_layer.h"

namespace ncv
{
//        /////////////////////////////////////////////////////////////////////////////////////////
//        // convolution (multi-layer) network model.
//        //
//        // parameters:
//        //      network - default = empty_string -> no hidden layer,
//        //              - format = [nconvs : crows : ccols : activation]*, where
//        //                          nconvs          - number of convolutions
//        //                          crows           - convolution size (rows)
//        //                          ccols           - convolution size (columns)
//        //                          activation      - activation function id
//        /////////////////////////////////////////////////////////////////////////////////////////

//        class conv_network_model_t : public model_t
//        {
//        public:
                
//                // constructor
//                conv_network_model_t(const string_t& params = string_t());

//                // create an object clone
//                virtual rmodel_t clone(const string_t& params) const
//                {
//                        return rmodel_t(new conv_network_model_t(params));
//                }

//                // compute the model output
//                virtual vector_t process(const tensor3d_t& input) const;

//        protected:

//                // save/load from file
//                virtual bool save(boost::archive::binary_oarchive& oa) const;
//                virtual bool load(boost::archive::binary_iarchive& ia);

//                // save/load from parameter vector
//                virtual bool save(vector_t& x) const;
//                virtual bool load(const vector_t& x);

//                // resize to new inputs/outputs, returns the number of parameters
//                virtual size_t resize();

//                // initialize parameters
//                virtual void zero();
//                virtual void random();

//                // construct the list of valid training samples
//                virtual void prune(data_t& data) const;

//                // compute loss value & gradient (given current
//                virtual scalar_t value(const data_t& data, const loss_t& loss) const;
//                virtual scalar_t vgrad(const data_t& data, const loss_t& loss, vector_t& grad) const;

//        private:

//                // process inputs (compute outputs & gradients)
//                const tensor3d_t& forward(const tensor3d_t& input) const;
//                void backward(const tensor3d_t& gradient) const;

//        private:

//                // attributes
//                conv_layers_t           m_network;              // convolution network
//                conv_network_params_t   m_params;               // parameters
//        };
}

#endif // NANOCV_MODEL_CONV_NETWORK_H
