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
                json_reader_t& config(json_reader_t& reader) final;
                json_writer_t& config(json_writer_t& writer) const final;
                solver_state_t minimize(const stoch_params_t&, const function_t&, const vector_t& x0) const final;

        private:

                // attributes
                scalar_t        m_alpha{100};
        };
}