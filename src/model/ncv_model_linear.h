#ifndef NANOCV_MODEL_LINEAR_H
#define NANOCV_MODEL_LINEAR_H

#include "ncv_model.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // linear model:
        //	output = weights * input + bias.
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class linear_model_t : public model_t
        {
        public:
                
                // constructor
                linear_model_t(const string_t& params = string_t());

                // create an object clone
                virtual rmodel_t clone(const string_t& params) const
                {
                        return rmodel_t(new linear_model_t(params));
                }

                // train the model
                virtual bool train(const task_t& task, const fold_t& fold, const loss_t& loss,
                                   size_t iters, scalar_t eps);

                // compute the model output
                virtual void process(const vector_t& input, vector_t& output) const;

                // save/load from file
                virtual bool save(const string_t& path) const;
                virtual bool load(const string_t& path);

                // access functions
                virtual size_t n_inputs() const { return m_weights.cols(); }
                virtual size_t n_outputs() const { return m_weights.rows(); }
                virtual size_t n_parameters() const { return m_weights.size() + m_bias.size(); }

        private:

                // resize to process new data
                void resize(size_t inputs, size_t outputs);

                // initialize parameters
                void initZero();
                void initRandom(scalar_t min, scalar_t max);

                // encode parameters for optimization
                virtual vector_t to_params() const;
                virtual void from_params(const vector_t& params);

        private:
                
                // attributes
                matrix_t		m_weights;
                vector_t		m_bias;
        };
}

#endif // NANOCV_MODEL_LINEAR_H
