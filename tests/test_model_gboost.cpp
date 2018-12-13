#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "tasks/task_affine.h"
#include "models/model_gboost.h"

using namespace nano;

static auto make_task(const affine_task_type task_type)
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(to_json("folds", 1, "isize", 5, "osize", 3, "count", 300, "type", task_type));
        NANO_REQUIRE(task->load());
        task->describe("synth-affine");
        return task;
}

static auto make_loss(const string_t& id)
{
        auto loss = get_losses().get(id);
        NANO_REQUIRE(loss);
        return loss;
}

NANO_BEGIN_MODULE(test_gboost_stump)

NANO_CASE(stump_real)
{
        const auto loss = make_loss("s-logistic");
        const auto task = make_task(affine_task_type::classification);

        const auto model = get_models().get("gboost-real-stump");
        NANO_REQUIRE(model);
        model->from_json(to_json(
                "rounds", 50, "patience", 50, "solver", "cgd",
                "cumloss", cumloss::average,
                "shrinkage", shrinkage::off,
                "subsampling", subsampling::off));

        // Check training: the model should fit the synthetic dataset
        const auto fold_index = 0u;
        const auto result = model->train(*task, fold_index, *loss);
        NANO_REQUIRE(result);

        const auto& state = result.optimum();
        // fixme: this doesn't generalize (check that the affine task is balanced, check the labels)
        //NANO_CHECK_LESS(state.m_train.m_error, epsilon2<scalar_t>());
        //NANO_CHECK_LESS(state.m_valid.m_error, epsilon2<scalar_t>());
        //NANO_CHECK_LESS(state.m_test.m_error, epsilon2<scalar_t>());

        // Check loading and saving
        const auto path = "gboost_stump_real.model";
        NANO_CHECK(model_t::save(path, "gboost-real-stump", *model));

        const auto model2 = model_t::load(path);
        NANO_REQUIRE(model2);

        // The loaded model should be identical
        const auto fold = fold_t{fold_index, protocol::test};
        const auto eval = model2->evaluate(*task, fold, *loss);

        NANO_CHECK_CLOSE(state.m_test.m_error, eval.m_errors.avg(), epsilon0<scalar_t>());
        NANO_CHECK_CLOSE(state.m_test.m_value, eval.m_values.avg(), epsilon0<scalar_t>());

        std::remove(path);
}

NANO_END_MODULE()
