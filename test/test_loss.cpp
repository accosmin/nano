#include "nano.h"
#include "class.h"
#include "utest.h"
#include "function.h"
#include "math/random.h"
#include "math/epsilon.h"

using namespace nano;

struct loss_function_t final : public function_t
{
        loss_function_t(const rloss_t& loss, const vector_t& target) :
                function_t("loss", target.size(), target.size(), target.size(), convexity::no, 1e+6),
                m_loss(loss), m_target(target)
        {
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                if (gx)
                {
                        *gx = m_loss->vgrad(m_target, x);
                }
                return m_loss->value(m_target, x);
        }

        const rloss_t&          m_loss;
        const vector_t&         m_target;
};

NANO_BEGIN_MODULE(test_loss)

NANO_CASE(evaluate)
{
        const strings_t loss_ids = get_losses().ids();

        const tensor_size_t cmd_min_dims = 2;
        const tensor_size_t cmd_max_dims = 10;
        const size_t cmd_tests = 128;

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const string_t& loss_id : loss_ids)
        {
                for (tensor_size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; ++ cmd_dims)
                {
                        const auto loss = get_losses().get(loss_id);
                        const auto target = class_target(cmd_dims / 2, cmd_dims);
                        const auto function = loss_function_t(loss, target);

                        random_t<scalar_t> rgen(scalar_t(-0.1), scalar_t(+0.1));

                        vector_t x(cmd_dims);

                        // check the gradient using random parameters
                        for (size_t t = 0; t < cmd_tests; ++ t)
                        {
                                rgen(x.data(), x.data() + x.size());

                                NANO_CHECK_GREATER(function.eval(x), 0.0);
                                NANO_CHECK_LESS(function.grad_accuracy(x), epsilon2<scalar_t>());

                                // loss value should upper-bound the error function
                                NANO_CHECK_GREATER(loss->value(x, target), loss->error(x, target) + epsilon0<scalar_t>());
                        }
                }
        }
}

NANO_END_MODULE()
