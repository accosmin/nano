#include "utest.h"
#include "solver_state.h"
#include "trainer_result.h"

using namespace nano;

const auto epochs = 200;
const auto best_epoch = 50;
const auto patience = size_t(32);
const auto optimum_error = static_cast<scalar_t>(epochs - best_epoch);

template <typename terror>
static trainer_state_t make_trainer_state(const terror valid_error, const size_t ms = 0, const size_t epoch = 0)
{
        const auto e = static_cast<scalar_t>(valid_error);
        return {milliseconds_t(ms), epoch, 0, 0,
                trainer_measurement_t{0, 0},
                trainer_measurement_t{0, e},
                trainer_measurement_t{0, 0}};
}

template <typename terror, typename tepoch>
static auto update_result(trainer_result_t& result, const opt_status status, const terror error, const tepoch epoch)
{
        solver_state_t opt_state;
        opt_state.m_status = status;

        return result.update(opt_state, make_trainer_state(error, 0, static_cast<size_t>(epoch)), patience);
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
                const auto error = i;

                const auto status = update_result(result, opt_status::max_iters, error, epoch);

                NANO_CHECK(status == trainer_status::better);
                NANO_CHECK(false == nano::is_done(status));
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_error, scalar_t(0));
        NANO_CHECK_EQUAL(result.optimum_epoch(), epochs);
}

NANO_CASE(result_solved)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto done = epoch >= best_epoch;
                const auto error = i;

                const auto status = update_result(result,
                        done ? opt_status::converged : opt_status::max_iters, error, epoch);

                NANO_CHECK((done ? trainer_status::solved : trainer_status::better) == status);
                NANO_CHECK((done ? true : false) == nano::is_done(status));

                if (done)
                {
                        break;
                }
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_error, optimum_error);
        NANO_CHECK_EQUAL(result.optimum_epoch(), best_epoch);
}

NANO_CASE(result_overfitting)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto error = (epoch <= best_epoch) ? i : (2 * (epochs - best_epoch) - i);

                const auto status = update_result(result, opt_status::max_iters, error, epoch);

                if (epoch <= best_epoch)
                {
                        NANO_CHECK(trainer_status::better == status);
                        NANO_CHECK(false == nano::is_done(status));
                }
                else if (epoch < best_epoch + static_cast<int>(patience))
                {
                        NANO_CHECK(trainer_status::worse == status);
                        NANO_CHECK(false == nano::is_done(status));
                }
                else
                {
                        NANO_CHECK(trainer_status::overfit == status);
                        NANO_CHECK(true == nano::is_done(status));
                        break;
                }
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_error, optimum_error);
        NANO_CHECK_EQUAL(result.optimum_epoch(), best_epoch);
}

NANO_CASE(result_not_finite)
{
        trainer_result_t result;
        for (int i = epochs; i >= 0; --i)
        {
                const auto epoch = epochs - i;
                const auto error = (epoch <= best_epoch) ? scalar_t(i) : scalar_t(NAN);

                const auto status = update_result(result, opt_status::max_iters, error, epoch);

                if (epoch <= best_epoch)
                {
                        NANO_CHECK(trainer_status::better == status);
                        NANO_CHECK(false == nano::is_done(status));
                }
                else
                {
                        NANO_CHECK(trainer_status::diverge == status);
                        NANO_CHECK(true == nano::is_done(status));
                        break;
                }
        }

        NANO_CHECK_EQUAL(result.optimum_state().m_valid.m_error, optimum_error);
        NANO_CHECK_EQUAL(result.optimum_epoch(), best_epoch);
}

NANO_END_MODULE()
