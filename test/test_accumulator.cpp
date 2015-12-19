#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "test_accumulator"

#include <boost/test/unit_test.hpp>
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "thread/thread.h"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/accumulator.h"

BOOST_AUTO_TEST_CASE(test_accumulator)
{
        using namespace cortex;

        cortex::init();

        const rtask_t task = cortex::get_tasks().get("random", "dims=2,rows=8,cols=8,color=luma,size=64");
        BOOST_REQUIRE_EQUAL(task.operator bool(), true);
        BOOST_CHECK_EQUAL(task->load(""), true);

        const samples_t samples = task->samples();
        const string_t cmd_model = "linear:dims=4;act-snorm;linear:dims=" + text::to_string(task->osize()) + ";";

        const rloss_t loss = cortex::get_losses().get("logistic");
        BOOST_CHECK_EQUAL(loss.operator bool(), true);

        const string_t criterion = "avg";
        const scalar_t lambda = 0.1;

        // create model
        const rmodel_t model = cortex::get_models().get("forward-network", cmd_model);
        BOOST_CHECK_EQUAL(model.operator bool(), true);
        BOOST_CHECK_EQUAL(model->resize(*task, true), true);

        model->random_params();

        // accumulators using 1 thread
        accumulator_t lacc(*model, 1, criterion, criterion_t::type::value, lambda);
        accumulator_t gacc(*model, 1, criterion, criterion_t::type::vgrad, lambda);

        BOOST_CHECK_EQUAL(lacc.lambda(), lambda);
        BOOST_CHECK_EQUAL(gacc.lambda(), lambda);

        lacc.set_lambda(lambda);
        gacc.set_lambda(lambda);

        BOOST_CHECK_EQUAL(lacc.lambda(), lambda);
        BOOST_CHECK_EQUAL(gacc.lambda(), lambda);

        lacc.update(*task, samples, *loss);
        const scalar_t value1 = lacc.value();

        BOOST_CHECK_EQUAL(lacc.count(), samples.size());

        gacc.update(*task, samples, *loss);
        const scalar_t vgrad1 = gacc.value();
        const vector_t pgrad1 = gacc.vgrad();

        BOOST_CHECK_EQUAL(gacc.count(), samples.size());
        BOOST_CHECK(std::isfinite(vgrad1));
        BOOST_CHECK_LE(math::abs(vgrad1 - value1), math::epsilon1<scalar_t>());

        // check results with multiple threads
        for (size_t nthreads = 2; nthreads < 3 * thread::n_threads(); ++ nthreads)
        {
                accumulator_t laccx(*model, nthreads, criterion, criterion_t::type::value, lambda);
                accumulator_t gaccx(*model, nthreads, criterion, criterion_t::type::vgrad, lambda);

                BOOST_CHECK_EQUAL(laccx.lambda(), lambda);
                BOOST_CHECK_EQUAL(gaccx.lambda(), lambda);

                laccx.set_lambda(lambda);
                gaccx.set_lambda(lambda);

                BOOST_CHECK_EQUAL(laccx.lambda(), lambda);
                BOOST_CHECK_EQUAL(gaccx.lambda(), lambda);

                laccx.update(*task, samples, *loss);

                BOOST_CHECK_EQUAL(laccx.count(), samples.size());
                BOOST_CHECK_LE(math::abs(laccx.value() - value1), math::epsilon1<scalar_t>());

                gaccx.update(*task, samples, *loss);

                BOOST_CHECK_EQUAL(gaccx.count(), samples.size());
                BOOST_CHECK_LE(math::abs(gaccx.value() - vgrad1), math::epsilon1<scalar_t>());
                BOOST_CHECK_LE((gaccx.vgrad() - pgrad1).lpNorm<Eigen::Infinity>(), math::epsilon1<scalar_t>());
        }
}
