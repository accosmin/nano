#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief initial step length
                ///
                /// \todo suitable for Newton and quasi-Netwon (LBFGS) methods
                ///
                ///
                template
                <
                        typename tstate,

                        typename tscalar = typename tstate::tscalar
                >
                class linesearch_init_unit_t
                {
                public:

                        tscalar update(const tstate&)
                        {
                                return tscalar(1);
                        }
                };
        }
}

