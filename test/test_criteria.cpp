#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_criteria"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/optimizer.h"
#include "cortex/accumulator.h"

BOOST_AUTO_TEST_CASE(test_criteria)
{
        using namespace cortex;

        cortex::init();

        const rtask_t task = cortex::get_tasks().get("random", "dims=2,rows=8,cols=8,color=luma,size=16");
        BOOST_REQUIRE_EQUAL(task.operator bool(), true);
        BOOST_CHECK_EQUAL(task->load(""), true);

        const samples_t samples = task->samples();
        const string_t cmd_model = "linear:dims=4;act-snorm;linear:dims=" + text::to_string(task->osize()) + ";";

        const rloss_t loss = cortex::get_losses().get("logistic");
        BOOST_REQUIRE_EQUAL(loss.operator bool(), true);

        // create model
        const rmodel_t model = cortex::get_models().get("forward-network", cmd_model);
        BOOST_REQUIRE_EQUAL(model.operator bool(), true);
        BOOST_CHECK_EQUAL(model->resize(*task, true), true);

        // vary criteria
        const strings_t ids = cortex::get_criteria().ids();
        for (const string_t& id : ids)
        {
                const rcriterion_t criterion = cortex::get_criteria().get(id);
                BOOST_REQUIRE_EQUAL(criterion.operator bool(), true);

                const scalar_t lambda = 0.1;
                const size_t nthreads = 1;

                accumulator_t lacc(*model, nthreads, *criterion, criterion_t::type::value, lambda);
                accumulator_t gacc(*model, nthreads, *criterion, criterion_t::type::vgrad, lambda);

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
                for (size_t t = 0; t < 3; ++ t)
                {
                        vector_t x;

                        model->random_params();
                        model->save_params(x);

                        BOOST_CHECK_GE(problem(x), 0.0);
                        BOOST_CHECK_LE(problem.grad_accuracy(x), math::epsilon1<scalar_t>());
                }
        }
}
