#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "tasks/task_affine.h"
#include "learners/gboost_stump.h"

using namespace nano;

static auto make_task_config()
{
        return to_json(
                "folds", 1, "isize", 6, "osize", 1, "noise", 0.0, "count", 1000,
                "type", affine_task_type::classification);
}

NANO_BEGIN_MODULE(test_gboost_stump)

NANO_CASE(stump_real)
{
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(make_task_config());
        NANO_REQUIRE(task->load());
        task->describe("synth-affine");

        const auto loss = get_losses().get("s-logistic");
        NANO_REQUIRE(loss);

        const auto learner = get_learners().get("gboost-stump");
        NANO_REQUIRE(learner);
        learner->from_json(to_json("rounds", 100, "patience", 10, "stump", stump_type::real));

        const auto result = learner->train(*task, 0u, *loss);
        NANO_REQUIRE(result);

        const auto& state = result.optimum();

        NANO_CHECK_LESS(state.m_train.m_value, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_valid.m_value, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_test.m_value, epsilon2<scalar_t>());

        NANO_CHECK_LESS(state.m_train.m_error, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_valid.m_error, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_test.m_error, epsilon2<scalar_t>());
}

NANO_CASE(stump_discrete)
{
        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(make_task_config());
        NANO_REQUIRE(task->load());

        const auto loss = get_losses().get("s-logistic");
        NANO_REQUIRE(loss);

        const auto learner = get_learners().get("gboost-stump");
        NANO_REQUIRE(learner);
        learner->from_json(to_json("rounds", 100, "patience", 10, "stump", stump_type::discrete));

        const auto result = learner->train(*task, 0u, *loss);
        NANO_REQUIRE(result);

        const auto& state = result.optimum();

        NANO_CHECK_LESS(state.m_train.m_value, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_valid.m_value, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_test.m_value, epsilon2<scalar_t>());

        NANO_CHECK_LESS(state.m_train.m_error, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_valid.m_error, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_test.m_error, epsilon2<scalar_t>());
}


NANO_END_MODULE()
