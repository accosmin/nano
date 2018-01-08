#include "text/table.h"
#include "math/stats.h"
#include "math/numeric.h"
#include "math/epsilon.h"
#include "solver_state.h"
#include "trainer_result.h"

using namespace nano;

trainer_result_t::trainer_result_t(const string_t& config) :
        m_config(config)
{
}

trainer_status trainer_result_t::update(const solver_state_t& opt_state, const trainer_state_t& state, const size_t patience)
{
        // out-of-bounds values (e.g. caused by invalid line-search steps)
        if (!state)
        {
                return trainer_status::diverge;
        }

        m_history.push_back(state);

        const auto updater = [&] ()
        {
                m_opt_params = opt_state.x;
                m_opt_state = state;
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

scalar_t trainer_result_t::convergence_speed() const
{
        const auto op = [](const trainer_state_t& prv, const trainer_state_t& crt)
        {
                assert(crt.m_train.m_value >= scalar_t(0));
                assert(prv.m_train.m_value >= scalar_t(0));
                assert(crt.m_milis >= prv.m_milis);

                const auto ratio_eps = nano::epsilon0<scalar_t>();
                const auto ratio = (ratio_eps + crt.m_train.m_value) / (ratio_eps + prv.m_train.m_value);
                const auto delta = 1 + crt.m_milis.count() - prv.m_milis.count();

                // convergence speed ~ loss decrease ratio / second
                const auto ret = static_cast<scalar_t>(std::pow(
                        static_cast<double>(ratio),
                        static_cast<double>(1000) / static_cast<double>(delta)));
                return std::isfinite(ret) ? nano::clamp(ret, scalar_t(0), scalar_t(1)) : scalar_t(1);
        };

        nano::stats_t<scalar_t> speeds;
        for (size_t i = 0; i + 1 < m_history.size(); ++ i)
        {
                speeds(op(m_history[i], m_history[i + 1]));
        }

        return static_cast<scalar_t>(speeds.avg());
}

bool trainer_result_t::save(const string_t& path) const
{
        table_t table;

        auto&& header = table.header();
        header  << "epoch"
                << "train_loss" << "train_error"
                << "valid_loss" << "valid_error"
                << "test_loss" << "test_error"
                << "seconds" << "xnorm" << "gnorm";

        size_t index = 0;
        for (const auto& state : history())
        {
                auto&& row = table.append();
                row     << (index ++)
                        << state.m_train.m_value << state.m_train.m_error
                        << state.m_valid.m_value << state.m_valid.m_error
                        << state.m_test.m_value << state.m_test.m_error
                        << idiv(state.m_milis.count(), 1000) << state.m_xnorm << state.m_gnorm;
        }

        return table.save(path, "    ");
}
bool nano::is_done(const trainer_status code)
{
        return  code == trainer_status::diverge ||
                code == trainer_status::overfit ||
                code == trainer_status::solved ||
                code == trainer_status::failed;
}
