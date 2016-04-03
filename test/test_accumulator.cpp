#include "unit_test.hpp"
#include "math/abs.hpp"
#include "cortex/cortex.h"
#include "thread/thread.h"
#include "math/epsilon.hpp"
#include "text/to_string.hpp"
#include "cortex/accumulator.h"
#include "cortex/layers/make_layers.h"

NANO_BEGIN_MODULE(test_accumulator)

NANO_CASE(evaluate)
{
        using namespace nano;

        nano::init();

        const auto task = nano::get_tasks().get("affine", "idims=1,irows=8,icols=8,osize=2,count=64");
        NANO_CHECK_EQUAL(task->load(), true);

        const auto cmd_model = make_affine_layer(4) + make_output_layer(task->osize());

        const auto fold = fold_t{0, protocol::train};
        const auto loss = nano::get_losses().get("logistic");

        const scalar_t lambda = 0.1;

        // create model
        const auto model = nano::get_models().get("forward-network", cmd_model);
        NANO_CHECK_EQUAL(model->resize(*task, true), true);

        model->random_params();

        // accumulators using 1 thread
        const auto criterion = nano::get_criteria().get("avg");

        accumulator_t lacc(*model, *loss, *criterion, criterion_t::type::value, lambda); lacc.set_threads(1);
        accumulator_t gacc(*model, *loss, *criterion, criterion_t::type::vgrad, lambda); gacc.set_threads(1);

        NANO_CHECK_EQUAL(lacc.lambda(), lambda);
        NANO_CHECK_EQUAL(gacc.lambda(), lambda);

        lacc.set_lambda(lambda);
        gacc.set_lambda(lambda);

        NANO_CHECK_EQUAL(lacc.lambda(), lambda);
        NANO_CHECK_EQUAL(gacc.lambda(), lambda);

        lacc.update(*task, fold);
        const scalar_t value1 = lacc.value();

        NANO_CHECK_EQUAL(lacc.count(), task->n_samples(fold));

        gacc.update(*task, fold);
        const scalar_t vgrad1 = gacc.value();
        const vector_t pgrad1 = gacc.vgrad();

        NANO_CHECK_EQUAL(gacc.count(), task->n_samples(fold));
        NANO_CHECK(std::isfinite(vgrad1));
        NANO_CHECK_CLOSE(vgrad1, value1, nano::epsilon1<scalar_t>());

        // check results with multiple threads
        for (size_t th = 2; th <= nano::n_threads(); ++ th)
        {
                accumulator_t laccx(*model, *loss, *criterion, criterion_t::type::value, lambda); laccx.set_threads(th);
                accumulator_t gaccx(*model, *loss, *criterion, criterion_t::type::vgrad, lambda); gaccx.set_threads(th);

                NANO_CHECK_EQUAL(laccx.lambda(), lambda);
                NANO_CHECK_EQUAL(gaccx.lambda(), lambda);

                laccx.set_lambda(lambda);
                gaccx.set_lambda(lambda);

                NANO_CHECK_EQUAL(laccx.lambda(), lambda);
                NANO_CHECK_EQUAL(gaccx.lambda(), lambda);

                laccx.update(*task, fold);

                NANO_CHECK_EQUAL(laccx.count(), task->n_samples(fold));
                NANO_CHECK_CLOSE(laccx.value(), value1, nano::epsilon1<scalar_t>());

                gaccx.update(*task, fold);

                NANO_CHECK_EQUAL(gaccx.count(), task->n_samples(fold));
                NANO_CHECK_CLOSE(gaccx.value(), vgrad1, nano::epsilon1<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(gaccx.vgrad(), pgrad1, nano::epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
