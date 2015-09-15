#pragma once

namespace ncv
{
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
                        void operator()(const tvector&, const tvector&, const tvector&, tsize&) const
                        {
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
                        void operator()(const tvector& gx, const tvector& crtx, const tvector& prvx, tsize& iter) const
                        {
                                if (gx.dot(crtx - prvx) > 0)
                                {
                                        iter = 1;
                                }
                        }
                };
        }
}

