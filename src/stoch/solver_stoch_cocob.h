#pragma once

#include "solver_stoch.h"

namespace nano
{
        ///
        /// \brief COCOB-Backprop,
        ///     see "Training Deep Networks without Learning Rates Through Coin Betting", by F. Orabona & T. Tommasi
        ///
        class stoch_cocob_t final : public stoch_solver_t
        {
        public:

                tuner_t configs() const final;
                void to_json(json_t&) const final;
                void from_json(const json_t&) final;

                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha{100};
                scalar_t        m_epsilon{static_cast<scalar_t>(1e-6)};
        };
}
