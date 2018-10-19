#include "utest.h"
#include "cortex.h"
#include "learners/gboost.h"

using namespace nano;

NANO_BEGIN_MODULE(test_gboost)

NANO_CASE(lsearch_function)
{
        tensor4d_t targets(4, 3, 2, 1);
        tensor4d_t soutputs(4, 3, 2, 1);
        tensor4d_t woutputs(4, 3, 2, 1);
        tensor4d_t outputs(4, 3, 2, 1);

        targets.vector(0) = class_target(0, 6);
        targets.vector(1) = class_target(1, 6);
        targets.vector(2) = class_target(2, 6);
        targets.vector(3) = class_target(3, 6);
        soutputs.random();
        woutputs.random();

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const auto& loss_id : get_losses().ids())
        {
                const auto loss = get_losses().get(loss_id);
                const auto func = gboost_lsearch_function_t{targets, soutputs, woutputs, outputs, *loss};

                for (auto i = 0; i < 13; ++ i)
                {
                        vector_t x = vector_t::Random(1);

                        NANO_CHECK_GREATER(func.vgrad(x), scalar_t(0));
                        NANO_CHECK_LESS(func.grad_accuracy(x), epsilon2<scalar_t>());
                }
        }
}

NANO_END_MODULE()