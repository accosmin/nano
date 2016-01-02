#include "unit_test.hpp"
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "thread/thread.h"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/accumulator.h"
#include "cortex/layers/make_layers.h"

NANOCV_BEGIN_MODULE(test_accumulator)

NANOCV_CASE(evaluate)
{
        using namespace cortex;

        cortex::init();

        const auto task = cortex::get_tasks().get("random", "dims=2,rows=8,cols=8,color=luma,size=64");
        NANOCV_CHECK_EQUAL(task->load(""), true);

        const samples_t samples = task->samples();
        const string_t cmd_model = make_affine_layer(4) + make_output_layer(task->osize());

        const auto loss = cortex::get_losses().get("logistic");

        const scalar_t lambda = 0.1;

        // create model
        const auto model = cortex::get_models().get("forward-network", cmd_model);
        NANOCV_CHECK_EQUAL(model->resize(*task, true), true);

        model->random_params();

        // accumulators using 1 thread
        const auto criterion = cortex::get_criteria().get("avg");

        accumulator_t lacc(*model, *criterion, criterion_t::type::value, lambda); lacc.set_threads(1);
        accumulator_t gacc(*model, *criterion, criterion_t::type::vgrad, lambda); gacc.set_threads(1);

        NANOCV_CHECK_EQUAL(lacc.lambda(), lambda);
        NANOCV_CHECK_EQUAL(gacc.lambda(), lambda);

        lacc.set_lambda(lambda);
        gacc.set_lambda(lambda);

        NANOCV_CHECK_EQUAL(lacc.lambda(), lambda);
        NANOCV_CHECK_EQUAL(gacc.lambda(), lambda);

        lacc.update(*task, samples, *loss);
        const scalar_t value1 = lacc.value();

        NANOCV_CHECK_EQUAL(lacc.count(), samples.size());

        gacc.update(*task, samples, *loss);
        const scalar_t vgrad1 = gacc.value();
        const vector_t pgrad1 = gacc.vgrad();

        NANOCV_CHECK_EQUAL(gacc.count(), samples.size());
        NANOCV_CHECK(std::isfinite(vgrad1));
        NANOCV_CHECK_CLOSE(vgrad1, value1, math::epsilon1<scalar_t>());

        // check results with multiple threads
        for (size_t nthreads = 2; nthreads <= thread::n_threads(); ++ nthreads)
        {
                accumulator_t laccx(*model, *criterion, criterion_t::type::value, lambda); laccx.set_threads(nthreads);
                accumulator_t gaccx(*model, *criterion, criterion_t::type::vgrad, lambda); gaccx.set_threads(nthreads);

                NANOCV_CHECK_EQUAL(laccx.lambda(), lambda);
                NANOCV_CHECK_EQUAL(gaccx.lambda(), lambda);

                laccx.set_lambda(lambda);
                gaccx.set_lambda(lambda);

                NANOCV_CHECK_EQUAL(laccx.lambda(), lambda);
                NANOCV_CHECK_EQUAL(gaccx.lambda(), lambda);

                laccx.update(*task, samples, *loss);

                NANOCV_CHECK_EQUAL(laccx.count(), samples.size());
                NANOCV_CHECK_CLOSE(laccx.value(), value1, math::epsilon1<scalar_t>());

                gaccx.update(*task, samples, *loss);

                NANOCV_CHECK_EQUAL(gaccx.count(), samples.size());
                NANOCV_CHECK_CLOSE(gaccx.value(), vgrad1, math::epsilon1<scalar_t>());
                NANOCV_CHECK_EIGEN_CLOSE(gaccx.vgrad(), pgrad1, math::epsilon1<scalar_t>());
        }
}

NANOCV_END_MODULE()
