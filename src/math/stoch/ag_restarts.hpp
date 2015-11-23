#pragma once

namespace math
{
        ///
        /// \brief restart strategy for Nesterov's accelerated gradient: no restart.
        ///
        template
        <
                typename tvector,
                typename tscalar,
                typename tsize
        >
        struct ag_no_restart_t
        {
                tsize operator()(
                        const tvector&,                                 ///< current function gradient
                        const tvector&, const tscalar,                  ///< current parameter & function value
                        const tvector&, const tscalar,                  ///< previous parameter & function value
                        const tsize iter) const
                {
                        return iter;
                }
        };

        ///
        /// \brief restart strategy for Nesterov's accelerated gradient: restart if function value increases.
        ///
        template
        <
                typename tvector,
                typename tscalar,
                typename tsize
        >
        struct ag_func_restart_t
        {
                tsize operator()(
                        const tvector&,                                 ///< current function gradient
                        const tvector&, const tscalar crtfx,            ///< current parameter & function value
                        const tvector&, const tscalar prvfx,            ///< previous parameter & function value
                        const tsize iter) const
                {
                        if (crtfx > prvfx)
                        {
                                return tsize(0);
                        }
                        else
                        {
                                return iter;
                        }
                }
        };

        ///
        /// \brief restart strategy for Nesterov's accelerated gradient: restart if not a descent direction.
        ///
        template
        <
                typename tvector,
                typename tscalar,
                typename tsize
        >
        struct ag_grad_restart_t
        {
                tsize operator()(
                        const tvector& crtgx,                           ///< current function gradient
                        const tvector& crtx, const tscalar,             ///< current parameter & function value
                        const tvector& prvx, const tscalar,             ///< previous parameter & function value
                        const tsize iter) const
                {
                        if (crtgx.dot(crtx - prvx) > 0)
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

