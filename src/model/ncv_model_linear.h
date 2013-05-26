#ifndef NANOCV_MODEL_LINEAR_H
#define NANOCV_MODEL_LINEAR_H

#include "ncv_model.h"

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////
        // linear model:
        //	output = weights * input + bias.
        //
        // parameters: iters=256[8-2048],eps=1e-5[1e-6,1e-3]
        //      iters   - number of optimization iterations (default = 256)
        //      eps     - optimization precision (default = 1e-5)
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
                virtual void process(const vector_t& input, vector_t& output) const;

                // save/load from file
                virtual bool save(const string_t& path) const;
                virtual bool load(const string_t& path);

                // access functions
                virtual size_t n_parameters() const { return m_weights.size() + m_bias.size(); }

        protected:

                // resize to new inputs/outputs
                virtual void resize();

                // initialize parameters
                virtual void zero();
                virtual void random();

                // train the model
                virtual bool train(const task_t& task, const samples_t& samples, const loss_t& loss);

        private:

                // encode parameters for optimization
                vector_t to_params() const;
                void from_params(const vector_t& params);

                // (partial) optimization data
                struct opt_data_t
                {
                        // constructor
                        opt_data_t(size_t n_outputs = 0, size_t n_inputs = 0)
                                :       m_fx(0.0),
                                        m_cnt(0)
                        {
                                resize(n_outputs, n_inputs);
                        }

                        // resize
                        void resize(size_t n_outputs, size_t n_inputs)
                        {
                                m_wgrad.resize(n_outputs, n_inputs);
                                m_bgrad.resize(n_outputs);
                                m_wgrad.setZero();
                                m_bgrad.setZero();
                        }

                        // cumulate (partial) results
                        void operator+=(const opt_data_t& data)
                        {
                                m_fx += data.m_fx;
                                m_cnt += data.m_cnt;
                                m_wgrad.noalias() += data.m_wgrad;
                                m_bgrad.noalias() += data.m_bgrad;
                        }

                        // attributes
                        scalar_t        m_fx;
                        size_t          m_cnt;
                        matrix_t        m_wgrad;
                        vector_t        m_bgrad;
                };

                // cumulate loss value and gradients
                void cum_fval(const task_t& task, const loss_t& loss, const sample_t& sample,
                        opt_data_t& data) const;

                void cum_fval_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
                        opt_data_t& data) const;

        private:
                
                // attributes
                matrix_t		m_weights;
                vector_t		m_bias;

                optimization_method     m_opt_method;   // optimization: method
                size_t                  m_opt_iters;    // optimization: maximum number of iterations
                size_t                  m_opt_eps;      // optimization: precision (epsilon)
        };
}

#endif // NANOCV_MODEL_LINEAR_H
