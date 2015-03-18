#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_loss"

#include <boost/test/unit_test.hpp>
#include "libnanocv/nanocv.h"
#include "libnanocv/util/random.hpp"
#include "libnanocv/math/epsilon.hpp"

namespace test
{
        using namespace ncv;

        void check_grad(const string_t& loss_id, size_t n_dims, size_t n_tests)
        {
                const rloss_t loss = loss_manager_t::instance().get(loss_id);

                const vector_t target = ncv::class_target(n_dims / 2, n_dims);

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
                        random_t<scalar_t> rgen(-1.0, +1.0);

                        vector_t x(n_dims);
                        rgen(x.data(), x.data() + n_dims);

                        BOOST_CHECK_GE(problem(x), 0.0);
                        BOOST_CHECK_LE(problem.grad_accuracy(x), math::epsilon1<scalar_t>());
                }
        }
}

BOOST_AUTO_TEST_CASE(test_loss)
{
        ncv::init();

        using namespace ncv;

        const strings_t loss_ids = loss_manager_t::instance().ids();

        const size_t cmd_min_dims = 2;
        const size_t cmd_max_dims = 10;
        const size_t cmd_tests = 128;

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const string_t& loss_id : loss_ids)
        {                
                for (size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; cmd_dims ++)
                {
                        test::check_grad(loss_id, cmd_dims, cmd_tests);
                }
        }
}
