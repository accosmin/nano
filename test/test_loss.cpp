#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "function.h"
#include "math/random.h"
#include "math/epsilon.h"
#include "tensor/numeric.h"

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
                NANO_CHECK_EQUAL(x.size(), m_targets.size());
                const auto scores = map_tensor(x.data(), m_targets.dims());

                if (gx)
                {
                        const auto grads = m_loss->vgrad(m_targets, scores);
                        NANO_CHECK_EQUAL(gx->size(), grads.size());
                        NANO_CHECK(nano::isfinite(grads));

                        *gx = grads.vector();
                }

                const auto values = m_loss->value(m_targets, scores);
                NANO_CHECK(nano::isfinite(values));
                return values.vector().sum();
        }

        const rloss_t&          m_loss;
        tensor4d_t              m_targets;
};

NANO_BEGIN_MODULE(test_loss)

NANO_CASE(gradient)
{
        const tensor_size_t cmd_min_dims = 2;
        const tensor_size_t cmd_max_dims = 10;
        const size_t cmd_tests = 128;

        // evaluate the analytical gradient vs. the finite difference approximation
        for (const auto& loss_id : get_losses().ids())
        {
                for (tensor_size_t cmd_dims = cmd_min_dims; cmd_dims <= cmd_max_dims; ++ cmd_dims)
                {
                        const auto loss = get_losses().get(loss_id);
                        const auto function = loss_function_t(loss, 3, cmd_dims);

                        for (size_t t = 0; t < cmd_tests; ++ t)
                        {
                                tensor1d_t x(3 * cmd_dims);
                                x.random(scalar_t(-0.1), scalar_t(+0.1));

                                NANO_CHECK_GREATER(function.eval(x.vector()), scalar_t(0));
                                NANO_CHECK_LESS(function.grad_accuracy(x.vector()), epsilon1<scalar_t>());
                        }
                }
        }
}

NANO_CASE(single_class)
{
        for (const auto& loss_id : {"classnll", "s-logistic", "s-exponential", "m-logistic", "m-exponential"})
        {
                const auto loss = get_losses().get(loss_id);
                NANO_REQUIRE(loss);

                const auto n_classes = 13;
                const auto i_label = 11;

                tensor4d_t target(1, n_classes, 1, 1);
                target.vector(0).setConstant(neg_target());
                target.vector(0)(i_label) = pos_target();

                tensor4d_t scores(1, n_classes, 1, 1);
                for (auto i = 0; i < n_classes; ++ i)
                {
                        scores.vector(0)(i) =
                                static_cast<scalar_t>(i + 1) *
                                (i == i_label ? pos_target() : neg_target());
                }

                const auto error = loss->error(target, scores);
                NANO_CHECK_LESS(error(0), epsilon0<scalar_t>());
        }
}

NANO_CASE(multi_class)
{
        for (const auto& loss_id : {"m-logistic", "m-exponential"})
        {
                const auto loss = get_losses().get(loss_id);
                NANO_REQUIRE(loss);

                const auto n_classes = 13;
                const auto i_label1 = 7, i_label2 = 11;

                tensor4d_t target(1, n_classes, 1, 1);
                target.vector(0).setConstant(neg_target());
                target.vector(0)(i_label1) = pos_target();
                target.vector(0)(i_label2) = pos_target();

                tensor4d_t scores(1, n_classes, 1, 1);
                for (auto i = 0; i < n_classes; ++ i)
                {
                        scores.vector(0)(i) =
                                static_cast<scalar_t>(i + 1) *
                                ((i == i_label1 || i == i_label2) ? pos_target() : neg_target());
                }

                const auto error = loss->error(target, scores);
                NANO_CHECK_LESS(error(0), epsilon0<scalar_t>());
        }
}

NANO_CASE(regression)
{
        for (const auto& loss_id : {"square", "cauchy"})
        {
                const auto loss = get_losses().get(loss_id);
                NANO_REQUIRE(loss);

                tensor4d_t target(3, 4, 1, 1);
                target.vector().setRandom();

                tensor4d_t scores = target;

                const auto error = loss->error(target, scores);
                NANO_CHECK_LESS(error(0), epsilon0<scalar_t>());
        }
}

NANO_END_MODULE()
