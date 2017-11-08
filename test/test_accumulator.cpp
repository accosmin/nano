#include "utest.h"
#include "accumulator.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "layers/make_layers.h"

using namespace nano;

NANO_BEGIN_MODULE(test_accumulator)

NANO_CASE(evaluate)
{
        const auto task = get_tasks().get("synth-charset", to_params("count", 64));
        NANO_CHECK(task->load());

        const auto fold = fold_t{0, protocol::train};
        const auto loss = nano::get_losses().get("s-logistic");

        // create model
        model_t model(make_affine_layer(4, 1, 1) + make_output_layer(task->odims()));

        NANO_CHECK(model.config(task->idims(), task->odims()));
        model.random();

        // accumulators using 1 thread
        accumulator_t acc(model, *loss);
        acc.threads(1);

        acc.mode(accumulator_t::type::value);
        acc.update(*task, fold);
        const auto value1 = acc.vstats().avg();

        NANO_CHECK_EQUAL(acc.vstats().count(), task->size(fold));

        acc.mode(accumulator_t::type::vgrad);
        acc.update(*task, fold);
        const auto vgrad1 = acc.vstats().avg();
        const auto pgrad1 = acc.vgrad();

        NANO_CHECK_EQUAL(acc.vstats().count(), task->size(fold));
        NANO_CHECK(std::isfinite(vgrad1));
        NANO_CHECK_CLOSE(vgrad1, value1, nano::epsilon1<scalar_t>());

        // check results with multiple threads
        for (size_t th = 2; th <= nano::logical_cpus(); ++ th)
        {
                accumulator_t accx(model, *loss);
                accx.threads(th);

                accx.mode(accumulator_t::type::value);
                accx.update(*task, fold);

                NANO_CHECK_EQUAL(accx.vstats().count(), task->size(fold));
                NANO_CHECK_CLOSE(accx.vstats().avg(), value1, nano::epsilon1<scalar_t>());

                accx.mode(accumulator_t::type::vgrad);
                accx.update(*task, fold);

                NANO_CHECK_EQUAL(accx.vstats().count(), task->size(fold));
                NANO_CHECK_CLOSE(accx.vstats().avg(), vgrad1, nano::epsilon1<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(accx.vgrad(), pgrad1, nano::epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
