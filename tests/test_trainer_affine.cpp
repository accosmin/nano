#include "utest.h"
#include "solver.h"
#include "builder.h"
#include "trainer.h"
#include "accumulator.h"
#include "core/epsilon.h"
#include "tasks/task_affine.h"
#include "trainers/function_batch.h"

using namespace nano;

NANO_BEGIN_MODULE(trainer_affine)

NANO_CASE(function)
{
        const auto isize = 3;
        const auto osize = 2;
        const auto fold = 0;

        // create synthetic task
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(to_json("isize", isize, "osize", osize, "noise", 0, "count", 100));
        NANO_REQUIRE(task->load());
        NANO_REQUIRE_EQUAL(task->idims(), make_dims(isize, 1, 1));
        NANO_REQUIRE_EQUAL(task->odims(), make_dims(osize, 1, 1));

        // create loss
        const auto loss = get_losses().get("square");
        NANO_REQUIRE(loss);

        // create model
        model_t model;
        NANO_REQUIRE(model.add(config_affine_node("fc", osize, 1, 1)));
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(make_dims(isize, 1, 1), make_dims(osize, 1, 1)));

        // check that the training function is valid
        accumulator_t acc(model, *loss);
        acc.mode(accumulator_t::type::vgrad);
        acc.threads(1);

        for (auto t = 0; t < 100; ++ t)
        {
                const auto x0 = vector_t::Random(model.psize());
                const auto function = batch_function_t{acc, *task, {fold, protocol::train}};

                NANO_CHECK_LESS(function.grad_accuracy(x0), epsilon2<scalar_t>());
        }
}

NANO_CASE(tune_and_train)
{
        const auto isize = 5;
        const auto osize = 3;
        const auto fold = 0;

        // create synthetic task
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(to_json("isize", isize, "osize", osize, "noise", 0, "count", 1000));
        NANO_REQUIRE(task->load());
        NANO_REQUIRE_EQUAL(task->idims(), make_dims(isize, 1, 1));
        NANO_REQUIRE_EQUAL(task->odims(), make_dims(osize, 1, 1));

        // create loss
        const auto loss = get_losses().get("square");
        NANO_REQUIRE(loss);

        // create model
        model_t model;
        NANO_REQUIRE(model.add(config_affine_node("fc", osize, 1, 1)));
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(make_dims(isize, 1, 1), make_dims(osize, 1, 1)));

        // create batch trainer
        const auto trainer = get_trainers().get("batch");
        NANO_REQUIRE(trainer);

        // check that the trainer works for all compatible solvers
        for (const auto& solver : get_solvers().ids())
        {
                trainer->from_json(to_json("epochs", 100, "solver", solver, "epsilon", epsilon2<scalar_t>()));

                accumulator_t acc(model, *loss);
                acc.threads(1);

                acc.random();
                const auto result = trainer->train(*task, fold, acc);
                NANO_REQUIRE(result);

                const auto state = result.optimum_state();
                NANO_CHECK_LESS(state.m_train.m_value, epsilon2<scalar_t>());
                NANO_CHECK_LESS(state.m_valid.m_value, epsilon2<scalar_t>());
                NANO_CHECK_LESS(state.m_test.m_value, epsilon2<scalar_t>());
        }
}

NANO_END_MODULE()
