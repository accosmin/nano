#include "utest.hpp"
#include "optim/state.h"
#include "trainer_result.h"

using namespace nano;

auto make_state(const state_t::status status)
{
        state_t state;
        state.m_status = status;
        return state;
}

auto make_trainer_state(const scalar_t valid_value, const size_t ms = 0, const size_t epoch = 0)
{
        return trainer_state_t(milliseconds_t(ms), epoch,
                trainer_measurement_t{0, 0, 0},
                trainer_measurement_t{valid_value, 0, 0},
                trainer_measurement_t{0, 0, 0});
}

NANO_BEGIN_MODULE(test_trainer)

NANO_CASE(state)
{
        const auto state1 = make_trainer_state(0);
        const auto state2 = make_trainer_state(1);
        const auto state3 = make_trainer_state(scalar_t(NAN));
        const auto state4 = make_trainer_state(scalar_t(INFINITY));

        NANO_CHECK_LESS(state1, state2);
        NANO_CHECK_LESS(state1, state3);
        NANO_CHECK_LESS(state1, state4);
}

NANO_CASE(result)
{
        trainer_result_t result;
        for (size_t i = 0; i < 100; ++ i)
        {

        }
}

NANO_END_MODULE()
