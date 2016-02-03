#pragma once

namespace math
{
        ///
        /// \brief keep track of the best state (useful for stochastic optimization methods)
        ///
        template
        <
                typename tstate
        >
        class best_state_t
        {
        public:

                ///
                /// \brief constructor
                ///
                explicit best_state_t(const tstate& state)
                        :       m_state(state)
                {
                }

                ///
                /// \brief update current state
                ///
                void update(const tstate& state)
                {
                        m_state.update(state);
                }

                ///
                /// \brief access the best state
                ///
                const tstate& get() const
                {
                        return m_state;
                }

        private:

                // attributes
                tstate          m_state;
        };
}

