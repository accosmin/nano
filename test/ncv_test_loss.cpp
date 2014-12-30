#include "nanocv.h"
#include <boost/program_options.hpp>
#include <set>

using namespace ncv;

static void test_grad(
        const string_t& header,
        const string_t& loss_id,
        size_t n_dims, size_t n_tests)
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

        // construct optimization problem: analytic gradient and finite difference approximation
        const opt_problem_t problem_gd(opt_fn_size, opt_fn_fval, opt_fn_grad);
        const opt_problem_t problem_ax(opt_fn_size, opt_fn_fval);

        for (size_t t = 0; t < n_tests; t ++)
        {
                random_t<scalar_t> rgen(-1.0, +1.0);

                vector_t x(n_dims);
                rgen(x.data(), x.data() + n_dims);

                vector_t gx_gd, gx_ax;
                problem_gd(x, gx_gd);
                problem_ax(x, gx_ax);

                const scalar_t dgx = (gx_gd - gx_ax).lpNorm<Eigen::Infinity>();

                log_info() << header << " [" << (t + 1) << "/" << n_tests
                           << "]: gradient accuracy = " << dgx << " (" << (dgx > 1e-8 ? "ERROR" : "OK") << ").";
        }
}

int main(int argc, char *argv[])
{
        ncv::init();

        const strings_t loss_ids = loss_manager_t::instance().ids();

        const size_t cmd_min_dims = 2;
        const size_t cmd_max_dims = 10;
        const size_t cmd_tests = 128;

        // evaluate the analytical gradient vs. the finite difference approximation
        //      for each: loss
        for (const string_t& loss_id : loss_ids)
        {                
                for (size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; cmd_dims ++)
                {
                        test_grad("[loss = " + loss_id + ", dims = " + text::to_string(cmd_dims) + "]",
                                  loss_id, cmd_dims, cmd_tests);
                }
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
