#include "unit_test.hpp"
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "cortex/accumulator.h"
#include "cortex/layers/make_layers.h"

NANO_BEGIN_MODULE(test_criteria)

NANO_CASE(evaluate)
{
        using namespace nano;

        nano::init();

        const auto task = nano::get_tasks().get("random", "dims=2,rows=5,cols=5,color=luma,size=16");
        NANO_CHECK_EQUAL(task->load(""), true);

        const samples_t samples = task->samples();
        const string_t cmd_model = make_affine_layer(3) + make_output_layer(task->osize());

        const auto loss = nano::get_losses().get("logistic");

        // create model
        const auto model = nano::get_models().get("forward-network", cmd_model);
        NANO_CHECK_EQUAL(model->resize(*task, true), true);

        // vary criteria
        const strings_t ids = nano::get_criteria().ids();
        for (const string_t& id : ids)
        {
                const auto criterion = nano::get_criteria().get(id);

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

                NANO_CHECK_GREATER(problem(x), 0.0);
                NANO_CHECK_LESS(problem.grad_accuracy(x), nano::epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
