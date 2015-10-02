#pragma once

namespace min
{
        ///
        /// \brief restart strategy for Nesterov's accelerated gradient: no restart.
        ///
        template
        <
                typename tvector,
                typename tsize
        >
        struct ag_no_restart_t
        {
                tsize operator()(const tvector&, const tvector&, const tvector&, const tsize iter) const
                {
                        return iter;
                }
        };

        ///
        /// \brief restart strategy for Nesterov's accelerated gradient: restart if bad direction.
        ///
        template
        <
                typename tvector,
                typename tsize
        >
        struct ag_grad_restart_t
        {
                tsize operator()(const tvector& gx, const tvector& crtx, const tvector& prvx, const tsize iter) const
                {
                        if (gx.dot(crtx - prvx) > 0)
                        {
                                return tsize(0);
                        }
                        else
                        {
                                return iter;
                        }
                }
        };
}

