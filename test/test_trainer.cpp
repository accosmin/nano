#include "utest.hpp"
#include "optim/state.h"
#include "trainer_result.h"
#include "text/to_string.hpp"

using namespace nano;

const int epochs = 200;
const int best_epoch = 50;

template <typename tvalue>
static auto make_trainer_state(const tvalue valid_value, const size_t ms = 0, const size_t epoch = 0)
{
        return trainer_state_t(milliseconds_t(ms), epoch,
                trainer_measurement_t{0, 0, 0},
                trainer_measurement_t{static_cast<scalar_t>(valid_value), 0, 0},
                trainer_measurement_t{0, 0, 0});
}

template <typename tvalue, typename tepoch>
static auto update_result(trainer_result_t& result, const opt_status status, const tvalue value, const tepoch epoch)
{
        state_t opt_state;
        opt_state.m_status = status;

        const auto config = trainer_config_t(1, {"param", scalar_t(0)});

        return result.update(opt_state, make_trainer_state(value, 0, static_cast<size_t>(epoch)), config);
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

NANO_CASE(result_max_iters)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto value = i;

                const auto status = update_result(result, opt_status::max_iters, value, epoch);

                NANO_CHECK(status == trainer_status::better);
                NANO_CHECK(false == nano::is_done(status, trainer_policy::stop_early));
                NANO_CHECK(false == nano::is_done(status, trainer_policy::all_epochs));
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_value, 0);
        NANO_CHECK_EQUAL(result.optimum_epoch(), epochs);
}

NANO_CASE(result_solved)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto done = epoch >= best_epoch;
                const auto value = i;

                const auto status = update_result(result,
                        done ? opt_status::converged : opt_status::max_iters, value, epoch);

                NANO_CHECK((done ? trainer_status::solved : trainer_status::better) == status);
                NANO_CHECK((done ? true : false) == nano::is_done(status, trainer_policy::stop_early));
                NANO_CHECK(false == nano::is_done(status, trainer_policy::all_epochs));

                if (done)
                {
                        break;
                }
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_value, epochs - best_epoch);
        NANO_CHECK_EQUAL(result.optimum_epoch(), best_epoch);
}

NANO_CASE(result_overfitting)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto value = (epoch <= best_epoch) ? i : (2 * (epochs - best_epoch) - i);

                const auto status = update_result(result, opt_status::max_iters, value, epoch);

                if (epoch <= best_epoch)
                {
                        NANO_CHECK(trainer_status::better == status);
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::stop_early));
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::all_epochs));
                }
                else if (epoch < best_epoch + static_cast<int>(trainer_result_t::overfitting_slack()))
                {
                        NANO_CHECK(trainer_status::worse == status);
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::stop_early));
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::all_epochs));
                }
                else
                {
                        NANO_CHECK(trainer_status::overfit == status);
                        NANO_CHECK(true == nano::is_done(status, trainer_policy::stop_early));
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::all_epochs));
                        break;
                }
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_value, epochs - best_epoch);
        NANO_CHECK_EQUAL(result.optimum_epoch(), best_epoch);
}

NANO_CASE(result_not_finite)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto value = (epoch <= best_epoch) ? scalar_t(i) : scalar_t(NAN);

                const auto status = update_result(result, opt_status::max_iters, value, epoch);

                if (epoch <= best_epoch)
                {
                        NANO_CHECK(trainer_status::better == status);
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::stop_early));
                        NANO_CHECK(false == nano::is_done(status, trainer_policy::all_epochs));
                }
                else
                {
                        NANO_CHECK(trainer_status::diverge == status);
                        NANO_CHECK(true == nano::is_done(status, trainer_policy::stop_early));
                        NANO_CHECK(true == nano::is_done(status, trainer_policy::all_epochs));
                        break;
                }
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_value, epochs - best_epoch);
        NANO_CHECK_EQUAL(result.optimum_epoch(), best_epoch);
}

NANO_END_MODULE()
