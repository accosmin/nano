#include "core/numeric.h"
#include "lsearch_morethuente.h"

using namespace nano;

///
/// see dcstep routine in MINPACK-2 (see http://ftp.mcs.anl.gov/pub/MINPACK-2/csrch/)
///
static void dcstep(
        scalar_t& stx, scalar_t& fx, scalar_t& dx,
        scalar_t& sty, scalar_t& fy, scalar_t& dy,
        scalar_t& stp, scalar_t& fp, scalar_t& dp,
        bool& brackt,
        const scalar_t stpmin, const scalar_t stpmax)
{
        scalar_t stpc, stpq, stpf;
        scalar_t theta, d1, d2, d3, s, gamma, p, q, r;

        const auto sgnd = dp * (dx / std::fabs(dx));

        if (fp > fx)
        {
	        theta = (fx - fp) * 3 / (stp - stx) + dx + dp;
                d1 = std::fabs(theta);
                d2 = std::fabs(dx);
                d1 = std::max(d1, d2);
                d2 = std::fabs(dp);
                s = std::max(d1, d2);
                d1 = theta / s;
                gamma = s * std::sqrt(d1 * d1 - dx / s * (dp / s));
                if (stp < stx)
                {
                        gamma = -gamma;
                }
                p = gamma - dx + theta;
                q = gamma - dx + gamma + dp;
                r = p / q;
                stpc = stx + r * (stp - stx);

                stpq = stx + dx / ((fx - fp) / (stp - stx) + dx) / 2 * (stp - stx);

                if (std::fabs(stpc - stx) < std::fabs(stpq - stx))
                {
                        stpf = stpc;
                }
                else
                {
                        stpf = stpc + (stpq - stpc) / 2;
                }
                brackt = true;
        }

        else if (sgnd < 0)
        {
                theta = (fx - fp) * 3 / (stp - stx) + dx + dp;
                d1 = std::fabs(theta);
                d2 = std::fabs(dx);
                d1 = std::max(d1, d2);
                d2 = std::fabs(dp);
                s = std::max(d1, d2);
                d1 = theta / s;
                gamma = s * std::sqrt(d1 * d1 - dx / s * (dp / s));
                if (stp > stx)
                {
                        gamma = -gamma;
                }
                p = gamma - dp + theta;
                q = gamma - dp + gamma + dx;
                r = p / q;
                stpc = stp + r * (stx - stp);

                stpq = stp + dp / (dp - dx) * (stx - stp);

                if (std::fabs(stpc - stp) > std::fabs(stpq - stp))
                {
                        stpf = stpc;
                }
                else
                {
                        stpf = stpq;
                }
                brackt = true;
        }

        else if (std::fabs(dp) < std::fabs(dx))
        {
                theta = (fx - fp) * 3 / (stp - stx) + dx + dp;
                d1 = std::fabs(theta);
                d2 = std::fabs(dx);
                d1 = std::max(d1, d2);
                d2 = std::fabs(dp);
                s = std::max(d1, d2);
                d3 = theta / s;
                d1 = 0;
                d2 = d3 * d3 - dx / s * (dp / s);
                gamma = s * std::sqrt(std::max(d1, d2));
                if (stp > stx)
                {
                        gamma = -gamma;
                }
                p = gamma - dp + theta;
                q = gamma + (dx - dp) + gamma;
                r = p / q;
                if (r < 0 && gamma != 0)
                {
                        stpc = stp + r * (stx - stp);
                }
                else if (stp > stx)
                {
                        stpc = stpmax;
                }
                else
                {
                        stpc = stpmin;
                }

                stpq = stp + dp / (dp - dx) * (stx - stp);

                if (brackt)
                {
                        if (std::fabs(stpc - stp) < std::fabs(stpq - stp))
                        {
                                stpf = stpc;
                        }
                        else
                        {
                                stpf = stpq;
                        }
                        if (stp > stx)
                        {
                                d1 = stp + (sty - stp) * .66;
                                stpf = std::min(d1, stpf);
                        }
                        else
                        {
                                d1 = stp + (sty - stp) * .66;
                                stpf = std::max(d1, stpf);
                        }
	        }
                else
                {
	                if (std::fabs(stpc - stp) > std::fabs(stpq - stp))
                        {
		                stpf = stpc;
                        }
                        else
                        {
                		stpf = stpq;
	                }
                	stpf = std::min(stpmax, stpf);
	                stpf = std::max(stpmin, stpf);
	        }
        }

        else
        {
	        if (brackt)
                {
                        theta = (fp - fy) * 3 / (sty - stp) + dy + dp;
                        d1 = std::fabs(theta);
                        d2 = std::fabs(dy);
                        d1 = std::max(d1, d2);
                        d2 = std::fabs(dp);
                        s = std::max(d1, d2);
                        d1 = theta / s;
                        gamma = s * sqrt(d1 * d1 - dy / s * (dp / s));
                        if (stp > sty)
                        {
                                gamma = -gamma;
                        }
                        p = gamma - dp + theta;
                        q = gamma - dp + gamma + dy;
                        r = p / q;
                        stpc = stp + r * (sty - stp);

                        stpf = stpc;
                }
                else if (stp > stx)
                {
                        stpf = stpmax;
                }
                else
                {
                        stpf = stpmin;
                }
        }


        if (fp > fx)
        {
                sty = stp;
	        fy = fp;
	        dy = dp;
        }
        else
        {
	        if (sgnd < 0.)
                {
	                sty = stx;
                        fy = fx;
                        dy = dx;
                }
                stx = stp;
                fx = fp;
                dx = dp;
        }

        stp = stpf;
}

lsearch_step_t lsearch_morethuente_t::get(const lsearch_step_t& step0, const scalar_t t0)
{
        const auto ftol = m_c1;
        const auto gtol = m_c2;
        const auto xtol = epsilon0<scalar_t>();

        const auto stpmin = lsearch_step_t::minimum();
        const auto stpmax = lsearch_step_t::maximum();

        lsearch_step_t step = step0;
        step.update(t0);

        int stage = 1;
        bool brackt = false;

        scalar_t stp = t0, f = step.phi(), g = step.gphi();
        scalar_t stmin = 0, stmax = stp + stp * 4;

        scalar_t width = stpmax - stpmin;
        scalar_t width1 = 2 * width;

        scalar_t finit = step.phi0(), ginit = step.gphi0(), gtest = ftol * ginit;
        scalar_t stx = 0, fx = finit, gx = ginit;
        scalar_t sty = 0, fy = finit, gy = ginit;

        for (auto i = 0; i < m_max_iterations; ++ i)
        {
                const auto ftest = finit + stp * gtest;
                if (stage == 1 && f <= ftest && g >= scalar_t(0))
                {
                        stage = 2;
                }

                // Check if further progress can be made
                if (brackt && (stp <= stmin || stp >= stmax))   return step;
                if (brackt && stmax - stmin <= xtol * stmax)    return step;
                if (stp == stpmax && f <= ftest && g <= gtest)  return step;
                if (stp == stpmin && (f > ftest || g >= gtest)) return step;

                // Check convergence
                if (f <= ftest && std::fabs(g) <= gtol * (-ginit))
                {
                        return step;
                }

                // Interpolate the next point to evaluate
                if (stage == 1 && f <= fx && f > ftest)
                {
                        auto fm = f - stp * gtest;
                        auto fxm = fx - stx * gtest;
                        auto fym = fy - sty * gtest;
                        auto gm = g - gtest;
                        auto gxm = gx - gtest;
                        auto gym = gy - gtest;

                        dcstep(stx, fxm, gxm, sty, fym, gym, stp, fm, gm, brackt, stmin, stmax);

                        fx = fxm + stx * gtest;
                        fy = fym + sty * gtest;
                        gx = gxm + gtest;
                        gy = gym + gtest;
                }
                else
                {
                        dcstep(stx, fx, gx, sty, fy, gy, stp, f, g, brackt, stmin, stmax);
                }

                // Decide if a bisection step is needed
                if (brackt)
                {
                        if (std::fabs(sty - stx) >= width1 * scalar_t(.66))
                        {
                                stp = stx + (sty - stx) * scalar_t(0.5);
                        }
                        width1 = width;
                        width = std::fabs(sty - stx);
                }

                // Set the minimum and maximum steps allowed for stp
                if (brackt)
                {
                        stmin = std::min(stx, sty);
                        stmax = std::max(stx, sty);
                }
                else
                {
                        stmin = stp + (stp - stx) * scalar_t(1.1);
                        stmax = stp + (stp - stx) * scalar_t(4.0);
                }

                // Force the step to be within the bounds stpmax and stpmin
                stp = nano::clamp(stp, stpmin, stpmax);

                // If further progress is not possible, let stp be the best point obtained during the search
                if (    (brackt && (stp <= stmin || stp >= stmax)) ||
                        (brackt && stmax - stmin <= xtol * stmax))
                {
                        stp = stx;
                }

                // Obtain another function and derivative
                step.update(stp);
                f = step.phi();
                g = step.gphi();
        }

        // NOK, give up
        return step0;
}
