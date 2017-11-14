#pragma once

#include "types.h"
#include "solver_state.h"

namespace nano
{
        ///
        /// \brief heuristics to initialize the step length,
        ///     see "Numerical optimization", Nocedal & Wright, 2nd edition, p.59
        ///
        class ls_init_t
        {
        public:
                ///
                /// \brief constructor
                ///
                explicit ls_init_t(const ls_initializer type);

                ///
                /// \brief compute the initial step length
                ///
                scalar_t operator()(const solver_state_t& cstate);

        private:

                ls_initializer  m_type;
                bool            m_first;        ///< check if first iteration
                scalar_t        m_prevf;        ///< previous function evaluation
                scalar_t        m_prevt0;       ///< previous step length
                scalar_t        m_prevdg;       ///< previous direction dot product
        };
}
