#include "unit_test.hpp"
#include "cortex/class.h"
#include "cortex/cortex.h"
#include "optim/problem.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"

using namespace nano;

static void check_grad(const string_t& loss_id, tensor_size_t n_dims, size_t n_tests)
{
        const auto loss = nano::get_losses().get(loss_id);

        const vector_t target = nano::class_target(n_dims / 2, n_dims);

        // optimization problem: size
        auto opt_fn_size = [&] ()
        {
                return n_dims;
        };

        // optimization problem: function value
        auto opt_fn_fval = [&] (const vector_t& x)
        {
                const vector_t& output = x;

                return loss->value(target, output);
        };

        // optimization problem: function value & gradient
        auto opt_fn_grad = [&] (const vector_t& x, vector_t& gx)
        {
                const vector_t& output = x;

                gx = loss->vgrad(target, output);

                return loss->value(target, output);
        };

        // construct optimization problem
        const problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_grad);

        // check the gradient using random parameters
        for (size_t t = 0; t < n_tests; ++ t)
        {
                nano::random_t<scalar_t> rgen(-1.0, +1.0);

                vector_t x(n_dims);
                rgen(x.data(), x.data() + n_dims);

                NANO_CHECK_GREATER(problem(x), 0.0);
                NANO_CHECK_LESS(problem.grad_accuracy(x), nano::epsilon2<scalar_t>());
        }
}

NANO_BEGIN_MODULE(test_loss)

NANO_CASE(evaluate)
{
        const strings_t loss_ids = nano::get_losses().ids();

        const tensor_size_t cmd_min_dims = 2;
        const tensor_size_t cmd_max_dims = 10;
        const size_t cmd_tests = 128;

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const string_t& loss_id : loss_ids)
        {
                for (tensor_size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; ++ cmd_dims)
                {
                        check_grad(loss_id, cmd_dims, cmd_tests);
                }
        }
}

NANO_END_MODULE()
