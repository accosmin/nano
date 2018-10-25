#include "core/table.h"
#include "core/stats.h"
#include "core/numeric.h"
#include "trainer_state.h"

using namespace nano;

trainer_result_t::trainer_result_t(string_t config) :
        m_config(std::move(config))
{
}

trainer_result_t::status trainer_result_t::update(const trainer_state_t& state, const int patience)
{
        // out-of-bounds values (e.g. caused by invalid line-search steps)
        if (!state)
        {
                return (m_status = status::diverge);
        }

        m_history.push_back(state);

        // improved performance
        if (state < m_optimum)
        {
                m_optimum = state;
                return (m_status = status::better);
        }

        // worse performance, but not enough epochs, keep training
        else if (state.m_epoch < patience)
        {
                return (m_status = status::worse);
        }

        // last improvement not far in the past, keep training
        else if (state.m_epoch < m_optimum.m_epoch + patience)
        {
                return (m_status = status::worse);
        }

        // no improvement since many epochs, overfitting detected
        else
        {
                return (m_status = status::overfit);
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

        nano::stats_t speeds;
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
                << "seconds";

        int index = 0;
        for (const auto& state : history())
        {
                auto&& row = table.append();
                row     << (index ++)
                        << state.m_train.m_value << state.m_train.m_error
                        << state.m_valid.m_value << state.m_valid.m_error
                        << state.m_test.m_value << state.m_test.m_error
                        << idiv(state.m_milis.count(), 1000);
        }

        return table.save(path);
}
