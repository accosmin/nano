#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "function.h"
#include "math/random.h"
#include "math/epsilon.h"

using namespace nano;

struct loss_function_t final : public function_t
{
        loss_function_t(const rloss_t& loss, const tensor_size_t count, const tensor_size_t xmaps) :
                function_t("loss", count * xmaps, count * xmaps, count * xmaps, convexity::no, 1e+6),
                m_loss(loss), m_targets(count, xmaps, 1, 1)
        {
                for (auto x = 0; x < count; ++ x)
                {
                        m_targets.vector(x) = class_target(x % xmaps, xmaps);
                }
        }

        scalar_t vgrad(const vector_t& x, vector_t* gx) const override
        {
                const auto scores = map_tensor(x.data(), m_targets.dims());
                if (gx)
                {
                        *gx = m_loss->vgrad(m_targets, scores).vector();
                }
                return m_loss->value(m_targets, scores).vector().sum();
        }

        const rloss_t&          m_loss;
        tensor4d_t              m_targets;
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
                        const auto function = loss_function_t(loss, 3, cmd_dims);

                        for (size_t t = 0; t < cmd_tests; ++ t)
                        {
                                tensor1d_t x(3 * cmd_dims);
                                x.random(scalar_t(-0.1), scalar_t(+0.1));

                                NANO_CHECK_GREATER(function.eval(x.vector()), 0);
                                NANO_CHECK_LESS(function.grad_accuracy(x.vector()), epsilon1<scalar_t>());
                        }
                }
        }
}

NANO_END_MODULE()
