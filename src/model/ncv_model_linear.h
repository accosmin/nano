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

                // compute the model output
                virtual const vector_t& process(const vector_t& input);

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

        private:
                
                // attributes
                matrix_t		m_weights;
                vector_t		m_bias;
                vector_t 		m_output;
        };
}

#endif // NANOCV_MODEL_LINEAR_H
