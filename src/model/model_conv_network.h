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
        //      network - network size:
        //                      - default = empty_string -> no hidden layer,
        //                      - format = [nconvs : crows : ccols : activation]*, where
        //                              nconvs          - number of convolutions
        //                              crows           - convolution size (rows)
        //                              ccols           - convolution size (columns)
        //                              activation      - activation function id
        //      iters   - number of optimization iterations (default = 256, options = [8, 4096])
        //      eps     - optimization convergence accuracy (default = 1e-5, options = [1e-3, 1e-6])
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
                scalar_t vgrad(const task_t&, const loss_t&, const samples_t&, vector_t&);

        private:

                // attributes
                conv_layers_t           m_layers;               // convolution network

                color_mode              m_color_param;          // input color mode
                conv_layer_params_t     m_layer_params;         // network description
                size_t                  m_opt_iters;
                scalar_t                m_opt_eps;
        };
}

#endif // NANOCV_MODEL_CONV_NETWORK_H
