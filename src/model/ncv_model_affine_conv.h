#ifndef NANOCV_MODEL_AFFINE_CONV_H
#define NANOCV_MODEL_AFFINE_CONV_H

#include "ncv_model.h"

namespace ncv
{
//        /////////////////////////////////////////////////////////////////////////////////////////
//        // affine convolution model:
//        //	output = input * conv + bias.
//        //
//        // parameters: iters=256[8-2048],eps=1e-5[1e-6,1e-3]
//        //      iters   - number of optimization iterations (default = 256)
//        //      eps     - optimization precision (default = 1e-5)
//        /////////////////////////////////////////////////////////////////////////////////////////
                
//        class affine_conv_model_t : public model_t
//        {
//        public:
                
//                // constructor
//                affine_conv_model_t(const string_t& params = string_t());

//                // create an object clone
//                virtual rmodel_t clone(const string_t& params) const
//                {
//                        return rmodel_t(new affine_conv_model_t(params));
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
//                struct data_t
//                {
//                        // constructor
//                        data_t(size_t n_outputs = 0, size_t n_rows = 0, size_t n_cols = 0);

//                        // resize
//                        void resize(size_t n_outputs, size_t n_rows, size_t n_cols);

//                        vector_t process(const matrix_t& input) const;
//                        void process(const matrix_t& input, const vector_t& target, const loss_t& loss);

//                        // cumulate (partial) results
//                        void operator+=(const data_t& data);

//                        // attributes
//                        scalar_t        m_loss;         // loss value
//                        size_t          m_count;        // #samples evaluated
//                        matrices_t      m_conv;         // convolution matrices
//                        matrices_t      m_gconv;        //      ... and their cumulated gradients
//                        vector_t        m_bias;         // bias (offset)
//                        vector_t        m_gbias;        //      ... and their cumulated gradients
//                };

//                // cumulate loss value and gradients
//                void cum_fval(const task_t& task, const loss_t& loss, const sample_t& sample, data_t& data) const;
//                void cum_fval_grad(const task_t& task, const loss_t& loss, const sample_t& sample, data_t& data) const;

//        private:
                
//                // attributes
//                matrices_t		m_conv;         // convolution matrices (for each output)
//                vector_t		m_bias;         // bias/offset

//                optimization_method     m_opt_method;   // optimization: method
//                size_t                  m_opt_iters;    // optimization: maximum number of iterations
//                size_t                  m_opt_eps;      // optimization: precision (epsilon)
//        };
}

#endif // NANOCV_MODEL_AFFINE_CONV_H
