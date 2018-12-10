#include "utest.h"
#include "models/stump.h"

using namespace nano;

NANO_BEGIN_MODULE(test_stump)

NANO_CASE(getset)
{
        stump_t stump;
        stump.feature(3);
        stump.threshold(scalar_t(-2.5));

        tensor4d_t outputs(2, 3, 1, 1);
        outputs.tensor(0).constant(-1);
        outputs.tensor(1).constant(+1);
        stump.outputs(outputs);

        NANO_CHECK_EQUAL(stump.feature(), 3);
        NANO_CHECK_EQUAL(stump.threshold(), scalar_t(-2.5));
        NANO_CHECK_EIGEN_CLOSE(stump.outputs().array(), outputs.array(), epsilon1<scalar_t>());
}

NANO_CASE(output)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, -2, +3);
        inputs.vector(1) = vector_t::LinSpaced(6, -3, +2);
        inputs.vector(2) = vector_t::LinSpaced(6, -4, +1);

        stump_t stump;
        stump.feature(0);
        stump.threshold(scalar_t(-2.5));

        tensor4d_t outputs(2, 3, 1, 1);
        outputs.tensor(0).constant(-1);
        outputs.tensor(1).constant(+1);
        stump.outputs(outputs);

        NANO_CHECK_EQUAL(stump.output(inputs.tensor(0)).size(), 3);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(1)).size(), 3);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(2)).size(), 3);

        NANO_CHECK_EQUAL(stump.output(inputs.tensor(0))(0), +1);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(1))(0), -1);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(2))(0), -1);
}

NANO_CASE(scale1)
{
        stump_t stump;
        stump.feature(0);
        stump.threshold(scalar_t(-3.5));

        tensor4d_t outputs(2, 3, 1, 1);
        outputs.tensor(0).constant(-1);
        outputs.tensor(1).constant(+1);
        stump.outputs(outputs);

        stump.scale(scalar_t(0.3));

        NANO_CHECK_EIGEN_CLOSE(stump.outputs().array(), outputs.array() * scalar_t(0.3), epsilon1<scalar_t>());
}

NANO_CASE(scalex)
{
        stump_t stump;
        stump.feature(0);
        stump.threshold(scalar_t(-3.5));

        tensor4d_t outputs(2, 3, 1, 1);
        outputs.tensor(0).constant(-1);
        outputs.tensor(1).constant(+1);
        stump.outputs(outputs);

        vector_t factors(3);
        factors(0) = 0.1;
        factors(1) = 0.2;
        factors(2) = 0.3;
        stump.scale(factors);

        NANO_CHECK_EIGEN_CLOSE(stump.outputs().array(0), outputs.array(0) * factors.array(), epsilon1<scalar_t>());
        NANO_CHECK_EIGEN_CLOSE(stump.outputs().array(1), outputs.array(1) * factors.array(), epsilon1<scalar_t>());
}

// todo: check stump_t::fit

NANO_END_MODULE()
