#pragma once

namespace ncv
{
        namespace optimize
        {
                ///
                /// \brief quadratic interpolation
                ///     (Nocedal & Wright (numerical optimization 2nd) @ p.50)
                ///
                template
                <
                        typename tstep
                >
                auto ls_quadratic_interpolation(const tstep& step1, const tstep& step2)
                        -> decltype(step1.m_step)
                {
                        return -(step1.m_grad * step2.m_step * step2.m_step) /
                                (2 * (step2.m_func - step1.m_func - step1.m_grad * step2.m_step));
                }

                /// \todo cubic interpolation!
        }
}

