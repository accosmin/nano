#include "utest.h"
#include "builder.h"
#include "accumulator.h"
#include "math/numeric.h"
#include "math/epsilon.h"

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
        const auto loss = get_losses().get("s-logistic");

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
        const auto value1 = acc.value();

        NANO_CHECK(std::isfinite(value1));
        NANO_CHECK_EQUAL(acc.vstats().count(), task->size(fold));
        NANO_CHECK_CLOSE(acc.vstats().avg(), value1, epsilon0<scalar_t>());

        acc.mode(accumulator_t::type::vgrad);
        acc.update(*task, fold);
        const auto vgrad1 = acc.value();
        const auto pgrad1 = acc.vgrad();

        NANO_CHECK(std::isfinite(vgrad1));
        NANO_CHECK_CLOSE(vgrad1, value1, epsilon0<scalar_t>());
        NANO_CHECK_EQUAL(acc.vstats().count(), task->size(fold));

        // check results with multiple threads
        for (size_t th = 2; th <= logical_cpus(); ++ th)
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
                        NANO_CHECK_CLOSE(accx.vstats().avg(), value1, epsilon0<scalar_t>());

                        accx.mode(accumulator_t::type::vgrad);
                        accx.update(*task, fold);

                        NANO_CHECK_EQUAL(accx.vstats().count(), task->size(fold));
                        NANO_CHECK_CLOSE(accx.vstats().avg(), vgrad1, epsilon0<scalar_t>());
                        NANO_CHECK_EIGEN_CLOSE(accx.vgrad(), pgrad1, epsilon0<scalar_t>());
                }
        }
}

NANO_CASE(regularization)
{
        const auto task = get_tasks().get("synth-charset");
        task->config(json_writer_t().object("count", 64).str());
        NANO_CHECK(task->load());

        const auto omaps = std::get<0>(task->odims());
        const auto orows = std::get<1>(task->odims());
        const auto ocols = std::get<2>(task->odims());
        const auto fold = fold_t{0, protocol::train};
        const auto loss = get_losses().get("s-logistic");

        // create model
        model_t model;
        NANO_CHECK(add_node(model, "1", affine_node_name(), config_affine_node, 4, 1, 1));
        NANO_CHECK(add_node(model, "2", "act-snorm", config_empty_node));
        NANO_CHECK(add_node(model, "3", affine_node_name(), config_affine_node, omaps, orows, ocols));
        NANO_CHECK(model.connect("1", "2", "3"));
        NANO_CHECK(model.done());

        NANO_CHECK(model.resize(task->idims(), task->odims()));
        model.random();

        // accumulators using no regularization
        accumulator_t acc(model, *loss);
        acc.threads(1);

        acc.mode(accumulator_t::type::vgrad);
        acc.update(*task, fold);
        const auto estats = acc.estats();
        const auto vstats = acc.vstats();
        const auto value = acc.value();
        const auto vgrad = acc.vgrad();

        NANO_CHECK_CLOSE(acc.value(), vstats.avg(), epsilon0<scalar_t>());

        // check L2 regularization
        for (const auto lambda : make_scalars(0, 0.000042, 0.00042, 0.0042, 0.042, 0.42))
        {
                acc.lambda(lambda);
                acc.update(*task, fold);

                NANO_CHECK_EQUAL(acc.estats(), estats);
                NANO_CHECK_EQUAL(acc.vstats(), vstats);

                NANO_CHECK_CLOSE(acc.value(), (value + lambda / 2 * acc.params().squaredNorm()), epsilon0<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(acc.vgrad(), (vgrad + lambda * acc.params()), epsilon0<scalar_t>());
        }
}

NANO_END_MODULE()
