#include "utest.h"
#include "builder.h"
#include "trainer.h"
#include "accumulator.h"
#include "math/epsilon.h"
#include "solver_stoch.h"

using namespace nano;

NANO_BEGIN_MODULE(trainer_stoch)

NANO_CASE(tune_and_train)
{
        const auto isize = 3;
        const auto osize = 2;
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
        const auto loss = get_losses().get("cauchy");
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
                if (    solver != "ag" &&
                        solver != "agfr" &&
                        solver != "aggr" &&
                        solver != "ngd" &&
                        solver != "adagrad" &&
                        solver != "adaratio")
                {
                        // todo: make the other solvers useful for this regression problem!
                        continue;
                }

                trainer->config(json_writer_t().object(
                        "tune_epochs", 8, "epochs", 1024, "batch", 16, "solver", solver,
                        "eps", epsilon1<scalar_t>()).str());

                accumulator_t acc(model, *loss);
                acc.threads(1);

                acc.random();
                trainer->tune(*enhancer, *task, fold, acc);

                acc.random();
                const auto result = trainer->train(*enhancer, *task, fold, acc);
                const auto state = result.optimum_state();

                NANO_CHECK(result);
                NANO_CHECK_LESS(state.m_train.m_value, epsilon2<scalar_t>());
                NANO_CHECK_LESS(state.m_train.m_error, epsilon3<scalar_t>());
        }
}

NANO_END_MODULE()
