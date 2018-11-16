#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "tasks/task_affine.h"
#include "models/gboost_stump.h"

using namespace nano;

NANO_BEGIN_MODULE(test_gboost_stump)

NANO_CASE(stump_real)
{
        const auto task_type = affine_task_type::classification;
        const auto stump_type = stump_type::real;

        const auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(to_json("folds", 1, "isize", 6, "osize", 1, "noise", 0.0, "count", 300, "type", task_type));
        NANO_REQUIRE(task->load());

        const auto loss = get_losses().get("s-logistic");
        NANO_REQUIRE(loss);

        const auto model = get_models().get("gboost-stump");
        NANO_REQUIRE(model);
        model->from_json(to_json("rounds", 100, "patience", 10, "stump", stump_type));

        // Check training: the model should fit the synthetic dataset
        const auto fold_index = 0u;
        const auto result = model->train(*task, fold_index, *loss);
        NANO_REQUIRE(result);

        const auto& state = result.optimum();
        NANO_CHECK_LESS(state.m_train.m_value, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_valid.m_value, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_test.m_value, epsilon2<scalar_t>());

        NANO_CHECK_LESS(state.m_train.m_error, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_valid.m_error, epsilon2<scalar_t>());
        NANO_CHECK_LESS(state.m_test.m_error, epsilon2<scalar_t>());

        // Check loading and saving
        const auto path = "gboost_stump_real.model";
        NANO_CHECK(model_t::save(path, "gboost-stump", *model));

        const auto model2 = model_t::load(path);
        NANO_REQUIRE(model2);

        // The loaded model should be identical
        const auto fold = fold_t{fold_index, protocol::test};
        const auto eval = model->evaluate(*task, fold, *loss);

        NANO_CHECK_CLOSE(state.m_test.m_error, eval.m_errors.avg(), epsilon0<scalar_t>());
        NANO_CHECK_CLOSE(state.m_test.m_value, eval.m_values.avg(), epsilon0<scalar_t>());

        std::remove(path);
}

NANO_END_MODULE()
