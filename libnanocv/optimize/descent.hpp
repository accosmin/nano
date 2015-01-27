#pragma once

#include <limits>

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief check and force a descent direction
                ///
                template
                <
                        typename tstate,
                        typename twlog,

                        typename tscalar = typename tstate::tscalar
                >
                tscalar descent(tstate& st, const twlog& wlog)
                {
                        tscalar dg = st.d.dot(st.g);
                        if (dg > std::numeric_limits<tscalar>::min())
                        {
                                if (wlog)
                                {
                                        wlog("not a descent direction!");
                                }
                                st.d = -st.g;
                                dg = st.d.dot(st.g);
                        }

                        return dg;
                }
        }
}

