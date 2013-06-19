#ifndef NANOCV_MODEL_CONV_NETWORK_H
#define NANOCV_MODEL_CONV_NETWORK_H

#include "model.h"
#include "tensor4d.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // convolution (multi-layer) network model.
        //
        // parameters:
        //      color   - input color mode (default = luma, options = luma/rgba)
        //      network - network size (default = empty_string -> no hidden layer,
        //                      format = [n_convolutions : n_convolution_rows x n_convolution_cols[,...]])
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class conv_network_model_t : public model_t
        {
        public:
                
                // constructor
                conv_network_model_t(const string_t& params = string_t());

                // create an object clone
                virtual rmodel_t clone(const string_t& params) const
                {
                        return rmodel_t(new conv_network_model_t(params));
                }

                // compute the model output
                virtual vector_t forward(const image_t& image, coord_t x, coord_t y) const;

        protected:

                // save/load from file
                virtual bool save(boost::archive::binary_oarchive& oa) const;
                virtual bool load(boost::archive::binary_iarchive& ia);

                // resize to new inputs/outputs, returns the number of parameters
                virtual size_t resize();

                // initialize parameters
                virtual void zero();
                virtual void random();

                // train the model
                virtual bool train(const task_t& task, const samples_t& samples, const loss_t& loss);

        private:

                typedef std::vector<tensor4d_t> conv_network_t;

                // compose the input data
                tensor3d_t make_input(const image_t& image, coord_t x, coord_t y) const;
                tensor3d_t make_input(const image_t& image, const rect_t& region) const;

                // retrieve the number of input channels
                size_t n_inputs() const;

                // encode parameters for optimization
                vector_t serialize() const;
                void deserialize(const vector_t& params);

                // cumulate loss value and gradients
                void cum_loss(const task_t&, const loss_t&, const sample_t&, conv_network_t&) const;
                void cum_grad(const task_t&, const loss_t&, const sample_t&, conv_network_t&) const;

        private:
                
                // attributes
                conv_network_t          m_layers;               // network layers
                color_mode              m_color_param;          // input color mode
                std::vector<size_t>     m_network_param;        // network description
        };
}

#endif // NANOCV_MODEL_CONV_NETWORK_H
