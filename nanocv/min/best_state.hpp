#pragma once

namespace min
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
                bool update(const tstate& state)
                {
                        if (state < m_state)
                        {
                                m_state = state;
                                return true;
                        }
                        else
                        {
                                return false;
                        }
                }

                const tstate& get() const
                {
                        return m_state;
                }

        private:

                // attributes
                tstate          m_state;
        };
}

