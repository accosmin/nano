#include "nano.h"
#include "utest.h"
#include "accumulator.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "layers/make_layers.h"

NANO_BEGIN_MODULE(test_accumulator)

NANO_CASE(evaluate)
{
        using namespace nano;

        const auto task = get_tasks().get("synth-charset", to_params("count", 64));

        NANO_CHECK_EQUAL(task->load(), true);

        const auto cmd_model = make_affine_layer(4) + make_output_layer(task->odims());

        const auto fold = fold_t{0, protocol::train};
        const auto loss = nano::get_losses().get("logistic");

        const scalar_t lambda = scalar_t(0.1);

        // create model
        const auto model = nano::get_models().get("forward-network", cmd_model);
        NANO_CHECK_EQUAL(model->configure(*task), true);

        model->random();

        // accumulators using 1 thread
        const auto criterion = nano::get_criteria().get("avg");

        accumulator_t acc(*model, *loss, *criterion);
        acc.lambda(lambda);
        acc.threads(1);

        NANO_CHECK_EQUAL(acc.lambda(), lambda);

        acc.mode(criterion_t::type::value);
        acc.update(*task, fold);
        const scalar_t value1 = acc.value();

        NANO_CHECK_EQUAL(acc.count(), task->n_samples(fold));

        acc.mode(criterion_t::type::vgrad);
        acc.update(*task, fold);
        const scalar_t vgrad1 = acc.value();
        const vector_t pgrad1 = acc.vgrad();

        NANO_CHECK_EQUAL(acc.count(), task->n_samples(fold));
        NANO_CHECK(std::isfinite(vgrad1));
        NANO_CHECK_CLOSE(vgrad1, value1, nano::epsilon1<scalar_t>());

        // check results with multiple threads
        for (size_t th = 2; th <= nano::logical_cpus(); ++ th)
        {
                accumulator_t accx(*model, *loss, *criterion);
                accx.lambda(lambda);
                accx.threads(th);

                NANO_CHECK_EQUAL(accx.lambda(), lambda);

                accx.mode(criterion_t::type::value);
                accx.update(*task, fold);

                NANO_CHECK_EQUAL(accx.count(), task->n_samples(fold));
                NANO_CHECK_CLOSE(accx.value(), value1, nano::epsilon1<scalar_t>());

                accx.mode(criterion_t::type::vgrad);
                accx.update(*task, fold);

                NANO_CHECK_EQUAL(accx.count(), task->n_samples(fold));
                NANO_CHECK_CLOSE(accx.value(), vgrad1, nano::epsilon1<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(accx.vgrad(), pgrad1, nano::epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
