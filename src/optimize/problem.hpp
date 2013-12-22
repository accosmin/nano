#ifndef NANOCV_OPTIMIZE_PROBLEM_HPP
#define NANOCV_OPTIMIZE_PROBLEM_HPP

#include "result.hpp"
#include <type_traits>
#include <functional>
#include <string>

namespace ncv
{
        namespace optimize
        {
                /////////////////////////////////////////////////////////////////////////////////////////
                // describes a multivariate optimization problem.
                /////////////////////////////////////////////////////////////////////////////////////////

                template
                <
                        typename tscalar_,
                        typename tsize_,
                        typename top_size,              // dimensionality operator: size = op()
                        typename top_fval,              // function value operator: f = op(x)
                        typename top_grad,              // function value and gradient operator: f = op(x, g)

                        // disable for not valid types!
                        typename tvalid_tscalar = typename std::enable_if<std::is_floating_point<tscalar_>::value>::type,
                        typename tvalid_tsize = typename std::enable_if<std::is_integral<tsize_>::value>::type
                >
                class problem_t
                {
                public:
                        typedef tscalar_                                                tscalar;
                        typedef tsize_                                                  tsize;

                        typedef typename tensor::vector_types_t<tscalar>::tvector       tvector;

                        // optimization final result
                        typedef result_t<tscalar, tsize>                                tresult;

                        // optimization current state
                        typedef typename tresult::tstate                                tstate;

                        // logging: warning, error, update (with the current state)
                        typedef std::function<void(const std::string&)>                 twlog;
                        typedef std::function<void(const std::string&)>                 telog;
                        typedef std::function<void(const tresult&)>                     tulog;

                        // constructor (analytic gradient)
                        explicit problem_t(
                                const top_size& op_size,
                                const top_fval& op_fval,
                                const top_grad& op_grad)
                                :       m_op_size(op_size),
                                        m_op_fval(op_fval),
                                        m_op_grad(op_grad),
                                        m_n_fvals(0),
                                        m_n_grads(0)
                        {
                        }

                        // constructor (no analytic gradient, can be estimated)
                        explicit problem_t(
                                const top_size& op_size,
                                const top_fval& op_fval)
                                :       problem_t(op_size, op_fval, top_grad())
                        {
                        }

                        // compute dimensionality & function value & gradient
                        tsize size() const { return _size(); }
                        tscalar operator()(const tvector& x) const { return _f(x); }
                        tscalar operator()(const tvector& x, tvector& g) const { return _f(x, g); }

                        // access functions
                        tsize n_fval_calls() const { return m_n_fvals; }
                        tsize n_grad_calls() const { return m_n_grads; }

                private:

                        // implementation: dimensionality
                        tsize _size() const
                        {
                                return m_op_size();
                        }

                        // implementation: function value
                        tscalar _f(const tvector& x) const
                        {
                                m_n_fvals ++;
                                return m_op_fval(x);
                        }

                        // implementation: function value & gradient
                        tscalar _f(const tvector& x, tvector& g) const
                        {
                                if (m_op_grad)
                                {
                                        m_n_fvals ++;
                                        m_n_grads ++;
                                        return m_op_grad(x, g);
                                }
                                else
                                {
                                        eval_grad(x, g);
                                        return _f(x);
                                }
                        }

                        // implementation: approximate gradient (if no analytic gradient provided)
                        void eval_grad(const tvector& x, tvector& g) const
                        {
                                const tsize n = size();
                                const tscalar d = 1e-6;//std::numeric_limits<tscalar>::epsilon();

                                tvector xp = x, xn = x;

                                g.resize(n);
                                for (tsize i = 0; i < n; i ++)
                                {
                                        if (i > 0)
                                        {
                                                xp(i - 1) -= d;
                                                xn(i - 1) += d;
                                        }

                                        xp(i) += d;
                                        xn(i) -= d;
                                        g(i) = _f(xp) - _f(xn);
                                }

                                g /= d * 2;
                        }

                private:

                        // attributes
                        top_size                m_op_size;
                        top_fval                m_op_fval;
                        top_grad                m_op_grad;
                        mutable tsize           m_n_fvals;              // #function value evaluations
                        mutable tsize           m_n_grads;              // #function gradient evaluations
                };
        }
}

#endif // NANOCV_OPTIMIZE_PROBLEM_HPP
