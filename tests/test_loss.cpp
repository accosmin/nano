#include "loss.h"
#include "utest.h"
#include "cortex.h"
#include "function.h"
#include "core/random.h"
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
                assert(x.size() == m_targets.size());
                const auto scores = map_tensor(x.data(), m_targets.dims());

                if (gx)
                {
                        const auto grads = m_loss->vgrad(m_targets, scores);
                        assert(gx->size() == grads.size());
                        assert(grads.array().isFinite().all());

                        *gx = grads.vector();
                }

                const auto values = m_loss->value(m_targets, scores);
                assert(values.array().isFinite().all());
                return values.vector().sum();
        }

        const rloss_t&          m_loss;
        tensor4d_t              m_targets;
};

NANO_BEGIN_MODULE(test_loss)

NANO_CASE(gradient)
{
        const tensor_size_t cmd_min_dims = 2;
        const tensor_size_t cmd_max_dims = 8;
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
                                NANO_CHECK_LESS(function.grad_accuracy(x.vector()), epsilon2<scalar_t>());
                        }
                }
        }
}

NANO_CASE(single_class)
{
        for (const auto& loss_id : {"classnll", "s-logistic", "s-exponential", "s-square", "s-cauchy"})
        {
                const auto loss = get_losses().get(loss_id);
                NANO_REQUIRE(loss);

                const auto n_classes = 13;
                tensor4d_t target(1, n_classes, 1, 1);
                tensor4d_t scores(1, n_classes, 1, 1);

                {
                        target.vector(0) = class_target(11, n_classes);
                        scores.vector(0) = class_target(11, n_classes);

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(0), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(11, n_classes);
                        scores.vector(0) = class_target(12, n_classes);

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(1), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(11, n_classes);
                        scores.vector(0) = class_target(11, n_classes);
                        scores.vector(0)(7) = pos_target() + 1;

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(1), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(11, n_classes);
                        scores.vector(0) = class_target(-1, n_classes);

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(1), epsilon0<scalar_t>());
                }
        }
}

NANO_CASE(multi_class)
{
        for (const auto& loss_id : {"m-logistic", "m-exponential", "m-square", "m-cauchy"})
        {
                const auto loss = get_losses().get(loss_id);
                NANO_REQUIRE(loss);

                const auto n_classes = 13;
                tensor4d_t target(1, n_classes, 1, 1);
                tensor4d_t scores(1, n_classes, 1, 1);

                {
                        target.vector(0) = class_target(n_classes);
                        scores.vector(0) = class_target(n_classes);

                        target.vector(0)(7) = target.vector(0)(9) = pos_target();
                        scores.vector(0)(7) = scores.vector(0)(9) = pos_target();

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(0), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(n_classes);
                        scores.vector(0) = class_target(n_classes);

                        target.vector(0)(7) = target.vector(0)(9) = pos_target();

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(2), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(n_classes);
                        scores.vector(0) = class_target(n_classes);

                        target.vector(0)(7) = target.vector(0)(9) = pos_target();
                        scores.vector(0)(5) = pos_target();

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(3), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(n_classes);
                        scores.vector(0) = class_target(n_classes);

                        target.vector(0)(7) = target.vector(0)(9) = pos_target();
                        scores.vector(0)(7) = pos_target();

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(1), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(n_classes);
                        scores.vector(0) = class_target(n_classes);

                        target.vector(0)(7) = target.vector(0)(9) = pos_target();
                        scores.vector(0)(5) = scores.vector(0)(9) = pos_target();

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(2), epsilon0<scalar_t>());
                }
                {
                        target.vector(0) = class_target(n_classes);
                        scores.vector(0) = class_target(n_classes);

                        target.vector(0)(7) = target.vector(0)(9) = pos_target();
                        scores.vector(0)(7) = scores.vector(0)(9) = scores.vector(0)(11) = pos_target();

                        const auto error = loss->error(target, scores);
                        NANO_CHECK_CLOSE(error(0), scalar_t(1), epsilon0<scalar_t>());
                }
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
