#include "unit_test.hpp"
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "cortex/accumulator.h"
#include "cortex/layers/make_layers.h"

NANOCV_BEGIN_MODULE(test_criteria)

NANOCV_CASE(evaluate)
{
        using namespace cortex;

        cortex::init();

        const auto task = cortex::get_tasks().get("random", "dims=2,rows=5,cols=5,color=luma,size=16");
        NANOCV_CHECK_EQUAL(task->load(""), true);

        const samples_t samples = task->samples();
        const string_t cmd_model = make_affine_layer(3) + make_output_layer(task->osize());

        const auto loss = cortex::get_losses().get("logistic");

        // create model
        const auto model = cortex::get_models().get("forward-network", cmd_model);
        NANOCV_CHECK_EQUAL(model->resize(*task, true), true);

        // vary criteria
        const strings_t ids = cortex::get_criteria().ids();
        for (const string_t& id : ids)
        {
                const auto criterion = cortex::get_criteria().get(id);

                const scalar_t lambda = 0.1;

                accumulator_t lacc(*model, *criterion, criterion_t::type::value, lambda);
                accumulator_t gacc(*model, *criterion, criterion_t::type::vgrad, lambda);

                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return lacc.psize();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        lacc.set_params(x);
                        lacc.update(*task, samples, *loss);
                        return lacc.value();
                };

                // optimization problem: function value & gradient
                auto opt_fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gacc.set_params(x);
                        gacc.update(*task, samples, *loss);
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                // construct optimization problem
                const opt_problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_grad);

                // check the gradient using random parameters
                vector_t x;
                model->random_params();
                model->save_params(x);

                NANOCV_CHECK_GREATER(problem(x), 0.0);
                NANOCV_CHECK_LESS(problem.grad_accuracy(x), math::epsilon1<scalar_t>());
        }
}

NANOCV_END_MODULE()
