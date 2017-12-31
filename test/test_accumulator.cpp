#include "utest.h"
#include "accumulator.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "layers/builder.h"

using namespace nano;

NANO_BEGIN_MODULE(test_accumulator)

NANO_CASE(evaluate)
{
        const auto task = get_tasks().get("synth-charset");
        task->config(json_writer_t().object("count", 64).str());
        NANO_CHECK(task->load());

        const auto omaps = std::get<0>(task->odims());
        const auto orows = std::get<1>(task->odims());
        const auto ocols = std::get<2>(task->odims());
        const auto fold = fold_t{0, protocol::train};
        const auto loss = nano::get_losses().get("s-logistic");

        // create model
        model_t model;
        NANO_CHECK(add_node(model, "1", affine_node_name(), config_affine_node, 4, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, omaps, orows, ocols));
        NANO_CHECK(model.connect("1", "2", "3"));
        NANO_CHECK(model.done());

        NANO_CHECK(model.resize(task->idims(), task->odims()));
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
                // and different minibatch sizes
                for (size_t bs = 2; bs <= 1024; bs *= 2)
                {
                        accumulator_t accx(model, *loss);
                        accx.mode(accumulator_t::type::value);
                        accx.minibatch(bs);
                        accx.threads(th);

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
}

NANO_END_MODULE()
