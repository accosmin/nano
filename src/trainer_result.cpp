#include "function_state.h"
#include "trainer_result.h"

using namespace nano;

trainer_status trainer_result_t::update(const function_state_t& opt_state,
        const trainer_state_t& state, const string_t& config, const size_t patience)
{
        // out-of-bounds values (e.g. caused by invalid line-search steps)
        if (!state)
        {
                return trainer_status::diverge;
        }

        m_history[config].push_back(state);

        const auto updater = [&] ()
        {
                m_opt_params = opt_state.x;
                m_opt_state = state;
                m_opt_config = config;
        };

        // optimization finished successfully
        if (opt_state.m_status == opt_status::converged)
        {
                if (state < m_opt_state)
                {
                        updater();
                }
                return trainer_status::solved;
        }

        // optimization failed
        else if (opt_state.m_status == opt_status::failed)
        {
                return trainer_status::failed;
        }

        // improved performance
        else if (state < m_opt_state)
        {
                updater();
                return trainer_status::better;
        }

        // worse performance, but not enough epochs, keep training
        else if (state.m_epoch < patience)
        {
                return trainer_status::worse;
        }

        // last improvement not far in the past, keep training
        else if (state.m_epoch < m_opt_state.m_epoch + patience)
        {
                return trainer_status::worse;
        }

        // no improvement since many epochs, overfitting detected
        else
        {
                return trainer_status::overfit;
        }
}

trainer_status trainer_result_t::update(const trainer_result_t& other)
{
        if (*this < other)
        {
                *this = other;
                return trainer_status::better;
        }

        else
        {
                return trainer_status::worse;
        }
}

trainer_state_t trainer_result_t::optimum_state() const
{
        return m_opt_state;
}

trainer_states_t trainer_result_t::optimum_states() const
{
        const auto it = m_history.find(m_opt_config);
        return (it == m_history.end()) ? trainer_states_t() : it->second;
}

vector_t trainer_result_t::optimum_params() const
{
        return m_opt_params;
}

string_t trainer_result_t::optimum_config() const
{
        return m_opt_config;
}

size_t trainer_result_t::optimum_epoch() const
{
        return optimum_state().m_epoch;
}

bool nano::operator<(const trainer_result_t& one, const trainer_result_t& other)
{
        return one.optimum_state() < other.optimum_state();
}

bool nano::is_done(const trainer_status code)
{
        return  code == trainer_status::diverge ||
                code == trainer_status::overfit ||
                code == trainer_status::solved ||
                code == trainer_status::failed;
}

std::ostream& nano::operator<<(std::ostream& os, const trainer_result_t& result)
{
        const auto state = result.optimum_state();

        os      << "train=" << state.m_train
                << ",valid=" << state.m_valid
                << ",test=" << state.m_test
                << "," << result.optimum_config() << ",epoch=" << result.optimum_epoch();
        if (result.optimum_states().size() > 1)
        {
                os << ",speed=" << convergence_speed(result.optimum_states()) << "/s";
        }
        else
        {
                os << ",speed=" << "0.0/s";
        }

        return os;
}
