#include "utest.h"
#include "builder.h"
#include "trainer.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "solver_stoch.h"
#include "tasks/task_affine.h"

using namespace nano;

NANO_BEGIN_MODULE(trainer_stoch)

NANO_CASE(tune_and_train_regression)
{
        const auto isize = 3;
        const auto osize = 2;
        const auto fold = 0;

        // create synthetic task
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object(
                "isize", isize, "osize", osize,
                "noise", 0, "count", 100, "type", affine_task_type::regression).str());
        NANO_REQUIRE(task->load());

        // create loss
        const auto loss = get_losses().get("square");
        NANO_REQUIRE(loss);

        // create model
        model_t model;
        NANO_REQUIRE(make_linear(model, osize, 1, 1, "act-snorm"));
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(make_dims(isize, 1, 1), make_dims(osize, 1, 1)));

        // create stochastic trainer
        const auto trainer = get_trainers().get("stoch");
        NANO_REQUIRE(trainer);

        // check that the trainer works for all compatible solvers
        for (const auto& solver : get_stoch_solvers().ids())
        {
                if (solver != "adam")
                {
                        // todo: have all stochastic solvers work properly!
                        continue;
                }

                trainer->config(json_writer_t().object(
                        "tune_epochs", 10, "epochs", 100, "batch", 1, "solver", solver,
                        "epsilon", epsilon1<scalar_t>()).str());

                accumulator_t acc(model, *loss);
                acc.threads(1);

                acc.random();
                trainer->tune(*task, fold, acc);

                acc.random();
                const auto result = trainer->train(*task, fold, acc);
                NANO_REQUIRE(result);

                const auto state = *result.history().rbegin();
                NANO_CHECK_LESS(state.m_train.m_value, epsilon1<scalar_t>());
                NANO_CHECK_LESS(state.m_train.m_error, epsilon2<scalar_t>());
        }
}

NANO_CASE(tune_and_train_classification)
{
        const auto isize = 3;
        const auto osize = 2;
        const auto fold = 0;

        // create synthetic task
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object(
                "isize", isize, "osize", osize,
                "noise", 0, "count", 1000, "type", affine_task_type::classification).str());
        NANO_REQUIRE(task->load());

        // create loss
        const auto loss = get_losses().get("m-logistic");
        NANO_REQUIRE(loss);

        // create model
        model_t model;
        NANO_REQUIRE(make_linear(model, osize, 1, 1, "act-snorm"));
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(make_dims(isize, 1, 1), make_dims(osize, 1, 1)));

        // create stochastic trainer
        const auto trainer = get_trainers().get("stoch");
        NANO_REQUIRE(trainer);

        // check that the trainer works for all compatible solvers
        for (const auto& solver : get_stoch_solvers().ids())
        {
                if (solver != "adam")
                {
                        // todo: have all stochastic solvers work properly!
                        continue;
                }

                trainer->config(json_writer_t().object(
                        "tune_epochs", 10, "epochs", 100, "batch", 1, "solver", solver,
                        "epsilon", epsilon1<scalar_t>()).str());

                accumulator_t acc(model, *loss);
                acc.threads(1);

                acc.random();
                trainer->tune(*task, fold, acc);

                acc.random();
                const auto result = trainer->train(*task, fold, acc);
                NANO_REQUIRE(result);

                const auto state = *result.history().rbegin();
                NANO_CHECK_LESS(state.m_train.m_error, epsilon3<scalar_t>());
        }
}

NANO_END_MODULE()
