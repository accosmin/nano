#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_loss"

#include <boost/test/unit_test.hpp>
#include "cortex/class.h"
#include "cortex/cortex.h"
#include "math/random.hpp"
#include "math/epsilon.hpp"
#include "cortex/optimizer.h"

namespace test
{
        using namespace cortex;

        void check_grad(const string_t& loss_id, tensor_size_t n_dims, size_t n_tests)
        {
                const rloss_t loss = cortex::get_losses().get(loss_id);

                const vector_t target = cortex::class_target(n_dims / 2, n_dims);

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
                const opt_problem_t problem(opt_fn_size, opt_fn_fval, opt_fn_grad);

                // check the gradient using random parameters
                for (size_t t = 0; t < n_tests; t ++)
                {
                        math::random_t<scalar_t> rgen(-1.0, +1.0);

                        vector_t x(n_dims);
                        rgen(x.data(), x.data() + n_dims);

                        BOOST_CHECK_GE(problem(x), 0.0);
                        BOOST_CHECK_LE(problem.grad_accuracy(x), math::epsilon1<scalar_t>());
                }
        }
}

BOOST_AUTO_TEST_CASE(test_loss)
{
        cortex::init();

        using namespace cortex;

        const strings_t loss_ids = cortex::get_losses().ids();

        const tensor_size_t cmd_min_dims = 2;
        const tensor_size_t cmd_max_dims = 10;
        const size_t cmd_tests = 128;

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const string_t& loss_id : loss_ids)
        {                
                for (tensor_size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; cmd_dims ++)
                {
                        test::check_grad(loss_id, cmd_dims, cmd_tests);
                }
        }
}
