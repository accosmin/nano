#include "nano.h"
#include "utest.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "math/numeric.h"
#include "layers/make_layers.h"

NANO_BEGIN_MODULE(test_criteria)

NANO_CASE(evaluate)
{
        using namespace nano;

        const auto task = get_tasks().get("affine", to_params("idims", 2, "irows", 5, "icols", 5, "osize", 2, "count", 20));
        NANO_CHECK_EQUAL(task->load(), true);

        const auto cmd_model = make_affine_layer(3) + make_output_layer(task->osize());
        const auto loss = get_losses().get("logistic");
        const auto fold = fold_t{0, protocol::train};
        const auto lambda = scalar_t(0.1);

        // create model
        const auto model = get_models().get("forward-network", cmd_model);
        NANO_CHECK_EQUAL(model->resize(*task, true), true);

        // vary criteria
        const strings_t ids = get_criteria().ids();
        for (const string_t& id : ids)
        {
                const auto criterion = get_criteria().get(id, "beta=1.0");

                accumulator_t lacc(*model, *loss, *criterion, criterion_t::type::value, lambda); lacc.set_threads(1);
                accumulator_t gacc(*model, *loss, *criterion, criterion_t::type::vgrad, lambda); gacc.set_threads(1);

                // optimization problem: size
                auto opt_fn_size = [&] ()
                {
                        return lacc.psize();
                };

                // optimization problem: function value
                auto opt_fn_fval = [&] (const vector_t& x)
                {
                        lacc.set_params(x);
                        lacc.update(*task, fold);
                        return lacc.value();
                };

                // optimization problem: function value & gradient
                auto opt_fn_grad = [&] (const vector_t& x, vector_t& gx)
                {
                        gacc.set_params(x);
                        gacc.update(*task, fold);
                        gx = gacc.vgrad();
                        return gacc.value();
                };

                // construct optimization problem
                const problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_grad);

                // check the gradient using random parameters
                vector_t x;
                model->random_params();
                model->save_params(x);

                NANO_CHECK_GREATER(problem(x), scalar_t(0));
                NANO_CHECK_LESS(problem.grad_accuracy(x), epsilon2<scalar_t>());
        }
}

NANO_END_MODULE()
