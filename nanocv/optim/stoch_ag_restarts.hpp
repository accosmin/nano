#pragma once

#include "nanocv/arch.h"

namespace ncv
{
        namespace optim
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
                        void operator()(const tvector& gx, const tvector& crtx, const tvector& prvx, tsize& iter) const
                        {
                                NANOCV_UNUSED1(gx);
                                NANOCV_UNUSED1(crtx);
                                NANOCV_UNUSED1(prvx);
                                NANOCV_UNUSED1(iter);
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

