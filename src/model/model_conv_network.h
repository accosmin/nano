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

                // compose the input data
                tensor3d_t make_input(const image_t& image, coord_t x, coord_t y) const;
                tensor3d_t make_input(const image_t& image, const rect_t& region) const;

                // access functions
                size_t n_inputs() const;
                size_t n_layers() const { return m_layers.size(); }
                conv_layer_t& layer(size_t l) { return m_layers[l]; }
                const conv_layer_t& layer(size_t l) const { return m_layers[l]; }

                // process inputs (compute outputs & gradients)
                const tensor3d_t& forward(const tensor3d_t& input) const;
                void backward(const tensor3d_t& gradient);

                // cumulate loss value and gradients
                scalar_t value(const task_t&, const loss_t&, const samples_t&);
                scalar_t vgrad(const task_t&, const loss_t&, const samples_t&);

        private:

                typedef std::vector<conv_layer_t>       layers_t;
                
                // attributes
                layers_t                m_layers;               // convolution network
                color_mode              m_color_param;          // input color mode
                std::vector<size_t>     m_network_param;        // network description
        };
}

#endif // NANOCV_MODEL_CONV_NETWORK_H
