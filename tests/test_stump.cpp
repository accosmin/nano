#include "utest.h"
#include "models/stump.h"

using namespace nano;

NANO_BEGIN_MODULE(test_stump)

NANO_CASE(test0)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, -2, +3);
        inputs.vector(1) = vector_t::LinSpaced(6, -3, +2);
        inputs.vector(2) = vector_t::LinSpaced(6, -4, +1);

        stump_t stump;
        stump.m_feature = 0;
        stump.m_threshold = scalar_t(-2.5);
        stump.m_outputs.resize(2, 3, 1, 1);
        stump.m_outputs.tensor(0).constant(-1);
        stump.m_outputs.tensor(1).constant(+1);

        NANO_CHECK_EQUAL(stump.output(inputs.tensor(0)).size(), 3);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(1)).size(), 3);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(2)).size(), 3);

        NANO_CHECK_EQUAL(stump.output(inputs.tensor(0))(0), +1);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(1))(0), -1);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(2))(0), -1);
}

NANO_CASE(test1)
{
        tensor4d_t inputs(3, 1, 2, 3);
        inputs.vector(0) = vector_t::LinSpaced(6, -2, +3);
        inputs.vector(1) = vector_t::LinSpaced(6, -3, +2);
        inputs.vector(2) = vector_t::LinSpaced(6, -4, +1);

        stump_t stump;
        stump.m_feature = 0;
        stump.m_threshold = scalar_t(-3.5);
        stump.m_outputs.resize(2, 3, 1, 1);
        stump.m_outputs.tensor(0).constant(-1);
        stump.m_outputs.tensor(1).constant(+1);

        NANO_CHECK_EQUAL(stump.output(inputs.tensor(0)).size(), 3);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(1)).size(), 3);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(2)).size(), 3);

        NANO_CHECK_EQUAL(stump.output(inputs.tensor(0))(0), +1);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(1))(0), +1);
        NANO_CHECK_EQUAL(stump.output(inputs.tensor(2))(0), -1);
}

NANO_END_MODULE()
