#include "utest.h"
#include "models/stump.h"
#include "models/gboost_loss_avg.h"
#include "models/gboost_loss_var.h"

using namespace nano;

static auto make_fold()
{
        return fold_t{size_t(0), protocol::train};
}

static auto make_task()
{
        auto task = get_tasks().get("synth-affine");
        NANO_REQUIRE(task);
        task->from_json(to_json("folds", 1, "isize", 3, "osize", 2, "count", 100));
        NANO_REQUIRE(task->load());
        return task;
}

static auto make_loss()
{
        auto loss = get_losses().get("square");
        NANO_REQUIRE(loss);
        return loss;
}

static auto make_stump()
{
        stump_t stump;
        stump.m_feature = 1;
        stump.m_threshold = 0;
        stump.m_outputs.resize(2, 2, 1, 1);
        stump.m_outputs.tensor(0).constant(-1);
        stump.m_outputs.tensor(1).constant(+1);
        return stump;
}

NANO_BEGIN_MODULE(test_gboost_loss)

NANO_CASE(avg_loss_stump_gradient)
{
        const auto task = make_task();
        const auto fold = make_fold();
        const auto loss = make_loss();
        const auto stump = make_stump();

        auto func = gboost_loss_avg_t<stump_t>{*task, fold, *loss};
        func.wlearner(stump);

        for (auto i = 0; i < 13; ++ i)
        {
                const vector_t x = vector_t::Random(1);
                NANO_CHECK_GREATER(func.vgrad(x), scalar_t(0));
                NANO_CHECK_LESS(func.grad_accuracy(x), epsilon2<scalar_t>());
        }
}

NANO_CASE(var_loss_stump_gradient)
{
        const auto task = make_task();
        const auto fold = make_fold();
        const auto loss = make_loss();
        const auto stump = make_stump();
        const auto lambda = scalar_t(0.1);

        auto func = gboost_loss_var_t<stump_t>{*task, fold, *loss, lambda};
        func.wlearner(stump);

        for (auto i = 0; i < 13; ++ i)
        {
                const vector_t x = vector_t::Random(1);
                NANO_CHECK_GREATER(func.vgrad(x), scalar_t(0));
                NANO_CHECK_LESS(func.grad_accuracy(x), epsilon2<scalar_t>());
        }
}

NANO_CASE(loss_stump_add_wlearner)
{
        const auto task = make_task();
        const auto fold = make_fold();
        const auto loss = make_loss();
        const auto stump = make_stump();

        auto func = gboost_loss_avg_t<stump_t>{*task, fold, *loss};

        const auto outputs0 = func.outputs();
        NANO_CHECK_CLOSE(outputs0.vector().minCoeff(), scalar_t(0), epsilon0<scalar_t>());
        NANO_CHECK_CLOSE(outputs0.vector().maxCoeff(), scalar_t(0), epsilon0<scalar_t>());

        func.add_wlearner(stump);

        const auto outputs1 = func.outputs();
        NANO_REQUIRE_EQUAL(outputs0.dims(), cat_dims(task->size(fold), task->odims()));
        NANO_REQUIRE_EQUAL(outputs1.dims(), cat_dims(task->size(fold), task->odims()));

        for (size_t i = 0, size = task->size(fold); i < size; ++ i)
        {
                NANO_CHECK_EIGEN_CLOSE(
                        outputs0.vector(i) + stump.output(task->input(fold, i)).vector(),
                        outputs1.vector(i),
                        epsilon0<scalar_t>());
        }
}

NANO_CASE(avg_loss_stump_update)
{
        const auto task = make_task();
        const auto fold = make_fold();
        const auto loss = make_loss();
        const auto stump = make_stump();

        auto func = gboost_loss_avg_t<stump_t>{*task, fold, *loss};
        func.add_wlearner(stump);
        const auto ret = func.update();

        const auto& outputs = func.outputs();
        const auto& residuals = func.residuals();

        scalar_t error = 0, value = 0;
        for (size_t i = 0, size = task->size(fold); i < size; ++ i)
        {
                const auto target = task->target(fold, i);
                const auto output = outputs.tensor(i);

                error += loss->error(target, output);
                value += loss->value(target, output);
                NANO_CHECK_EIGEN_CLOSE(
                        residuals.vector(i),
                        loss->vgrad(target, output).vector(),
                        epsilon0<scalar_t>());
        }

        NANO_CHECK_CLOSE(ret.first, value / static_cast<scalar_t>(task->size(fold)), epsilon0<scalar_t>());
        NANO_CHECK_CLOSE(ret.second, error / static_cast<scalar_t>(task->size(fold)), epsilon0<scalar_t>());
}

// todo: check if ::vgrad(x) is computing the correct overall loss
// todo: check if ::update() is returning the correct overall loss and the correct average error

// todo: check if ::vgrad(x) is computing the correct overall loss
// todo: check if ::update() is returning the correct overall loss and the correct average error

NANO_END_MODULE()
