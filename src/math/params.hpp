#pragma once

namespace math
{
        ///
        /// \brief common parameters for batch optimization
        ///
        template
        <
                typename tproblem                       ///< optimization problem
        >
        class params_t
        {
        public:

                using tsize = typename tproblem::tsize;
                using tstate = typename tproblem::tstate;
                using tscalar = typename tproblem::tscalar;
                using tvector = typename tproblem::tvector;
                using topulog = typename tproblem::topulog;

                ///
                /// \brief constructor
                ///
                params_t(const topulog& u = topulog())
                        :       m_ulog(u)
                {
                }

                ///
                /// \brief destructor
                ///
                virtual ~params_t()
                {
                }

                ///
                /// \brief log current optimization state
                ///
                bool ulog(const tstate& state) const
                {
                        return m_ulog ? m_ulog(state) : true;
                }

                topulog         m_ulog;         ///< update log: (the current_state_after_each_epoch)
        };
}

