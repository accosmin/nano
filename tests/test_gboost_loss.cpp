#include "utest.h"
#include "models/stump.h"
#include "models/gboost_loss_avg.h"
#include "models/gboost_loss_var.h"

using namespace nano;

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
        const auto loss = make_loss();
        const auto stump = make_stump();

        // evaluate the analytical gradient vs. the finite difference approximation
        auto func = gboost_loss_avg_t<stump_t>{*task, fold_t{size_t(0), protocol::train}, *loss};
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
        const auto loss = make_loss();
        const auto stump = make_stump();
        const auto lambda = scalar_t(0.1);

        // evaluate the analytical gradient vs. the finite difference approximation
        auto func = gboost_loss_var_t<stump_t>{*task, fold_t{size_t(0), protocol::train}, *loss, lambda};
        func.wlearner(stump);

        for (auto i = 0; i < 13; ++ i)
        {
                const vector_t x = vector_t::Random(1);
                NANO_CHECK_GREATER(func.vgrad(x), scalar_t(0));
                NANO_CHECK_LESS(func.grad_accuracy(x), epsilon2<scalar_t>());
        }
}

/*
NANO_CASE(lsearch_evaluation)
{
        tensor4d_t targets(4, 3, 2, 1);
        tensor4d_t soutputs(4, 3, 2, 1);
        tensor4d_t woutputs(4, 3, 2, 1);

        targets.vector(0) = class_target(6, 0);
        targets.vector(1) = class_target(6, 1);
        targets.vector(2) = class_target(6, 2);
        targets.vector(3) = class_target(6, 3);

        soutputs.random();
        woutputs.random();

        // verify the function value against the loss value
        for (const auto& loss_id : get_losses().ids())
        {
                const auto loss = get_losses().get(loss_id);
                const auto func = gboost_lsearch_function_t{targets, soutputs, woutputs, *loss};

                for (auto i = 0; i < 13; ++ i)
                {
                        vector_t x = vector_t::Random(1);

                        tensor4d_t outputs(4, 3, 2, 1);
                        outputs.vector() = soutputs.vector() + x(0) * woutputs.vector();

                        NANO_CHECK_CLOSE(
                                func.vgrad(x) * 4,
                                loss->value(targets.tensor(0), outputs.tensor(0)) +
                                loss->value(targets.tensor(1), outputs.tensor(1)) +
                                loss->value(targets.tensor(2), outputs.tensor(2)) +
                                loss->value(targets.tensor(3), outputs.tensor(3)),
                                epsilon0<scalar_t>());
                }
        }
}*/

NANO_END_MODULE()
