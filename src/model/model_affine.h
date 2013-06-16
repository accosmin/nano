#ifndef NANOCV_MODEL_AFFINE_H
#define NANOCV_MODEL_AFFINE_H

#include "model.h"
#include "olayer.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // affine convolution model.
        //
        // parameters:
        //      proc    - input processing method (default = luma, range = luma/rgba)
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class affine_model_t : public model_t
        {
        public:
                
                // constructor
                affine_model_t(const string_t& params = string_t());

                // create an object clone
                virtual rmodel_t clone(const string_t& params) const
                {
                        return rmodel_t(new affine_model_t(params));
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
                matrices_t make_input(const image_t& image, coord_t x, coord_t y) const;
                matrices_t make_input(const image_t& image, const rect_t& region) const;

                // retrieve the number of input channels
                size_t n_inputs() const;

                // encode parameters for optimization
                vector_t serialize() const;
                void deserialize(const vector_t& params);

                // cumulate loss value and gradients
                void cum_loss(const task_t& task, const loss_t& loss, const sample_t& sample, olayer_t& data) const;
                void cum_grad(const task_t& task, const loss_t& loss, const sample_t& sample, olayer_t& data) const;

        private:
                
                // attributes
                olayer_t                m_olayer;       //
                ncv::process            m_opt_proc;     // input processing method
        };
}

#endif // NANOCV_MODEL_AFFINE_H
