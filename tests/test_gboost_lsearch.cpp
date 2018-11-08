#include "utest.h"
#include "cortex.h"
#include "models/gboost_lsearch.h"

using namespace nano;

NANO_BEGIN_MODULE(test_gboost_lsearch)

NANO_CASE(lsearch_gradient)
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

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const auto& loss_id : get_losses().ids())
        {
                const auto loss = get_losses().get(loss_id);
                const auto func = gboost_lsearch_function_t{targets, soutputs, woutputs, *loss};

                for (auto i = 0; i < 13; ++ i)
                {
                        vector_t x = vector_t::Random(1);

                        NANO_CHECK_GREATER(func.vgrad(x), scalar_t(0));
                        NANO_CHECK_LESS(func.grad_accuracy(x), epsilon2<scalar_t>());
                }
        }
}

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
}

NANO_END_MODULE()
