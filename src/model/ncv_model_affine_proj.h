#ifndef NANOCV_MODEL_AFFINE_PROJ_H
#define NANOCV_MODEL_AFFINE_PROJ_H

#include "ncv_model.h"

namespace ncv
{
//        /////////////////////////////////////////////////////////////////////////////////////////
//        // affine projective model:
//        //	output = U * input * V * p + bias.
//        //
//        // parameters: iters=256[8-2048],eps=1e-5[1e-6,1e-3]
//        //      iters   - number of optimization iterations (default = 256)
//        //      eps     - optimization precision (default = 1e-5)
//        /////////////////////////////////////////////////////////////////////////////////////////
                
//        class affine_proj_model_t : public model_t
//        {
//        public:
                
//                // constructor
//                affine_proj_model_t(const string_t& params = string_t());

//                // create an object clone
//                virtual rmodel_t clone(const string_t& params) const
//                {
//                        return rmodel_t(new affine_proj_model_t(params));
//                }

//                // compute the model output
//                virtual vector_t process(const image_t& image, coord_t x, coord_t y) const;

//        protected:

//                // save/load from file
//                virtual bool save(std::ofstream& os) const;
//                virtual bool load(std::ifstream& is);

//                // resize to new inputs/outputs, returns the number of parameters
//                virtual size_t resize();

//                // initialize parameters
//                virtual void zero();
//                virtual void random();

//                // train the model
//                virtual bool train(const task_t& task, const samples_t& samples, const loss_t& loss);

//        private:

//                // encode parameters for optimization
//                vector_t to_params() const;
//                void from_params(const vector_t& params);

//                // (partial) optimization data
//                struct opt_data_t
//                {
//                        // constructor
//                        opt_data_t(size_t n_outputs = 0, size_t n_rows = 0, size_t n_cols = 0)
//                                :       m_fx(0.0),
//                                        m_cnt(0)
//                        {
//                                resize(n_outputs, n_rows, n_cols);
//                        }

//                        // resize
//                        void resize(size_t n_outputs, size_t n_rows, size_t n_cols)
//                        {
//                                m_ugrad.resize(n_outputs, n_rows);
//                                m_vgrad.resize(n_cols, n_outputs);
//                                m_pgrad.resize(n_outputs);
//                                m_bgrad.resize(n_outputs);

//                                m_ugrad.setZero();
//                                m_vgrad.setZero();
//                                m_pgrad.setZero();
//                                m_bgrad.setZero();
//                        }

//                        // cumulate (partial) results
//                        void operator+=(const opt_data_t& data)
//                        {
//                                m_fx += data.m_fx;
//                                m_cnt += data.m_cnt;
//                                m_ugrad.noalias() += data.m_ugrad;
//                                m_vgrad.noalias() += data.m_vgrad;
//                                m_pgrad.noalias() += data.m_pgrad;
//                                m_bgrad.noalias() += data.m_bgrad;
//                        }

//                        // attributes
//                        scalar_t        m_fx;
//                        size_t          m_cnt;
//                        matrix_t        m_ugrad;
//                        matrix_t        m_vgrad;
//                        vector_t        m_pgrad;
//                        vector_t        m_bgrad;
//                };

//                // cumulate loss value and gradients
//                void cum_fval(const task_t& task, const loss_t& loss, const sample_t& sample,
//                        opt_data_t& data) const;

//                void cum_fval_grad(const task_t& task, const loss_t& loss, const sample_t& sample,
//                        opt_data_t& data) const;

//        private:
                
//                // attributes
//                matrix_t		m_u;
//                matrix_t                m_v;
//                vector_t                m_p;
//                vector_t		m_b;

//                optimization_method     m_opt_method;   // optimization: method
//                size_t                  m_opt_iters;    // optimization: maximum number of iterations
//                size_t                  m_opt_eps;      // optimization: precision (epsilon)
//        };
}

#endif // NANOCV_MODEL_AFFINE_PROJ_H
