#include "utest.h"
#include "builder.h"
#include "trainer.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "solver_batch.h"

using namespace nano;

NANO_BEGIN_MODULE(trainer_batch)

NANO_CASE(tune_and_train)
{
        const auto isize = 5;
        const auto osize = 3;
        const auto fold = 0;

        // create synthetic task
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->config(json_writer_t().object("isize", isize, "osize", osize, "noise", 0, "count", 100).str());
        NANO_REQUIRE(task->load());

        // create default enhancer (use task as it is)
        const auto enhancer = get_enhancers().get("default");
        NANO_REQUIRE(enhancer);

        // create loss
        const auto loss = get_losses().get("square");
        NANO_REQUIRE(loss);

        // create model
        model_t model;
        NANO_REQUIRE(make_linear(model, osize, 1, 1, "act-snorm"));
        NANO_REQUIRE(model.done());
        NANO_REQUIRE(model.resize(make_dims(isize, 1, 1), make_dims(osize, 1, 1)));

        // create batch trainer
        const auto trainer = get_trainers().get("batch");
        NANO_REQUIRE(trainer);

        // check that the trainer works for all compatible solvers
        for (const auto& solver : get_batch_solvers().ids())
        {
                trainer->config(json_writer_t().object(
                        "epochs", 128, "solver", solver, "eps", epsilon0<scalar_t>()).str());

                accumulator_t acc(model, *loss);
                acc.threads(1);

                acc.random();
                trainer->tune(*enhancer, *task, fold, acc);

                acc.random();
                const auto result = trainer->train(*enhancer, *task, fold, acc);
                const auto state = result.optimum_state();

                NANO_CHECK(result);
                NANO_CHECK_LESS(state.m_train.m_value, epsilon1<scalar_t>());
                NANO_CHECK_LESS(state.m_train.m_error, epsilon1<scalar_t>());
        }
}

NANO_END_MODULE()
