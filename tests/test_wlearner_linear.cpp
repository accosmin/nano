#include "utest.h"
#include "core/ibstream.h"
#include "core/obstream.h"
#include "models/wlearner_linear.h"

using namespace nano;

static auto get_a()
{
        tensor3d_t a(3, 1, 1);
        a.constant(-1);
        return a;
}

static auto get_b()
{
        tensor3d_t b(3, 1, 1);
        b.constant(+1);
        return b;
}

const auto a = get_a();
const auto b = get_b();

NANO_BEGIN_MODULE(test_model_linear)

NANO_CASE(getset)
{
        wlearner_linear_t learner;
        learner.feature(2);
        learner.a(a);
        learner.b(b);

        NANO_CHECK_EQUAL(learner.feature(), 2);
        NANO_CHECK_EIGEN_CLOSE(learner.a().array(), a.array(), epsilon0<scalar_t>());
        NANO_CHECK_EIGEN_CLOSE(learner.b().array(), b.array(), epsilon0<scalar_t>());
}

NANO_CASE(output)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, -2, +3);
        inputs.vector(1) = vector_t::LinSpaced(6, -3, +2);
        inputs.vector(2) = vector_t::LinSpaced(6, -4, +1);

        wlearner_linear_t learner;
        learner.feature(0);
        learner.a(a);
        learner.b(b);

        NANO_CHECK_EQUAL(learner.output(inputs.tensor(0)).size(), 3);
        NANO_CHECK_EQUAL(learner.output(inputs.tensor(1)).size(), 3);
        NANO_CHECK_EQUAL(learner.output(inputs.tensor(2)).size(), 3);

        NANO_CHECK_EQUAL(learner.output(inputs.tensor(0))(0), +3);
        NANO_CHECK_EQUAL(learner.output(inputs.tensor(1))(0), +4);
        NANO_CHECK_EQUAL(learner.output(inputs.tensor(2))(0), +5);
}

NANO_CASE(scale1)
{
        wlearner_linear_t learner;
        learner.feature(0);
        learner.a(a);
        learner.b(b);

        learner.scale(scalar_t(0.3));

        NANO_CHECK_EIGEN_CLOSE(learner.a().array(), a.array() * scalar_t(0.3), epsilon0<scalar_t>());
        NANO_CHECK_EIGEN_CLOSE(learner.b().array(), b.array() * scalar_t(0.3), epsilon0<scalar_t>());
}

NANO_CASE(scalex)
{
        wlearner_linear_t learner;
        learner.feature(0);
        learner.a(a);
        learner.b(b);

        vector_t factors(3);
        factors(0) = 0.1;
        factors(1) = 0.2;
        factors(2) = 0.3;
        learner.scale(factors);

        NANO_CHECK_EIGEN_CLOSE(learner.a().array(), a.array() * factors.array(), epsilon0<scalar_t>());
        NANO_CHECK_EIGEN_CLOSE(learner.b().array(), b.array() * factors.array(), epsilon0<scalar_t>());
}

NANO_CASE(serialize)
{
        wlearner_linear_t learner;
        learner.feature(4);
        learner.a(a);
        learner.b(b);

        const auto path_learner = "learner.linear";
        {
                obstream_t ostream(path_learner);
                NANO_CHECK(learner.save(ostream));
        }
        {
                wlearner_linear_t learner2;

                ibstream_t istream(path_learner);
                NANO_CHECK(learner2.load(istream));

                NANO_CHECK_EQUAL(learner.feature(), learner2.feature());
                NANO_CHECK_EIGEN_CLOSE(learner.a().vector(), learner2.a().vector(), epsilon0<scalar_t>());
                NANO_CHECK_EIGEN_CLOSE(learner.b().vector(), learner2.b().vector(), epsilon0<scalar_t>());
        }

        std::remove(path_learner);
}

// todo: check fitting

NANO_END_MODULE()
