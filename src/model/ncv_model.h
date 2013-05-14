#ifndef NANOCV_MODEL_H
#define NANOCV_MODEL_H

#include "ncv_task.h"
#include "ncv_loss.h"
#include "ncv_optimize.h"

namespace ncv
{
        // manage models (register new ones, query and clone them)
        class model_t;
        typedef manager_t<model_t>              model_manager_t;
        typedef model_manager_t::robject_t      rmodel_t;

        /////////////////////////////////////////////////////////////////////////////////////////
        // generic model: output = model(input).
        /////////////////////////////////////////////////////////////////////////////////////////
                
        class model_t : public clonable_t<model_t>
        {
        public:

                // constructor
                model_t(const string_t& name, const string_t& description)
                        :       clonable_t<model_t>(name, description)
                {
                }

                // destructor
                virtual ~model_t() {}

                // train the model
                virtual bool train(const task_t& task, const fold_t& fold, const loss_t& loss,
                                   size_t iters, scalar_t eps) = 0;

                // evaluate the model (compute the average loss value & error)
                void test(const task_t& task, const fold_t& fold, const loss_t& loss,
                        scalar_t& lvalue, scalar_t& lerror) const;

                // compute the model output
                virtual void process(const vector_t& input, vector_t& output) const = 0;

                // save/load from file
                virtual bool save(const string_t& path) const = 0;
                virtual bool load(const string_t& path) = 0;

                // access functions
                virtual size_t n_inputs() const = 0;
                virtual size_t n_outputs() const = 0;
                virtual size_t n_parameters() const = 0;

        protected:

                // encode parameters for optimization
                virtual vector_t to_params() const = 0;
                virtual void from_params(const vector_t& params) = 0;

        public:

                // encode parameters/gradients for optimization
                static void encode(const matrix_t& mat, size_t& pos, vector_t& params);
                static void encode(const vector_t& vec, size_t& pos, vector_t& params);

                static void decode(matrix_t& mat, size_t& pos, const vector_t& params);
                static void decode(vector_t& vec, size_t& pos, const vector_t& params);

        protected:

                // optimization problem
                typedef std::function<size_t(void)>                                     opt_size_t;
                typedef std::function<scalar_t(const vector_t&)>                        opt_fval_t;
                typedef std::function<scalar_t(const vector_t&, vector_t&)>             opt_fval_grad_t;
                typedef optimize::problem_t<opt_size_t, opt_fval_t, opt_fval_grad_t>    opt_problem_t;
        };
}

#endif // NANOCV_MODEL_H
