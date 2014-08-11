#ifndef NANOCV_SEARCH1D_H
#define NANOCV_SEARCH1D_H

#include <cmath>
#include <limits>
#include <map>
#include "convolution.hpp"

namespace ncv
{
        namespace detail 
        {
                template
                <
                        typename toperator,
                        typename tscalar
                >
                tscalar update_history(const toperator& op, tscalar param, std::map<tscalar, tscalar>& history)
                {
                        typename std::map<tscalar, tscalar>::iterator it = history.find(param);
                        if (it == history.end())
                        {
                                return history[param] = op(std::exp(param));
                        }
                        else
                        {
                                return it->second;
                        }
                }
        }
        
        ///
        /// \brief search for a 1D parameter that minimizes a given operator, 
        ///     using a greedy approach on the logarithmic scale.
        ///
        /// \returns the optimum (log) parameter
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns a tscalar score
                typename tscalar
        >
        tscalar min_search1d(const toperator& op, tscalar minlog, tscalar maxlog, tscalar epslog)
        {
                std::map<tscalar, tscalar> history;
                
                tscalar bestlog = (maxlog + minlog) / 2;
                
                for (   tscalar distlog = (maxlog - minlog) / 2;
                        (distlog > epslog && epslog > tscalar(0);
                        distlog /= 2)
                {
                        const tscalar param1 = bestlog - distlog;
                        const tscalar param2 = bestlog;
                        const tscalar param3 = bestlog + distlog;
                        
                        const tscalar score1 = detail::update_history(op, param1, history);
                        const tscalar score2 = detail::update_history(op, param2, history);
                        const tscalar score3 = detail::update_history(op, param3, history);
                        
                        if (score1 < score2 && score1 < score3)
                        {
                                bestlog = param1;
                        }
                        else if (score1 < score2 && score1 < score3)
                        {
                                bestlog = param1;
                        }
                        else if (score1 < score2 && score1 < score3)
                        {
                                bestlog = param1;
                        }
                        else
                        {
                                break;
                        }
                }
                
                return  history.empty() ? 
                        std::numeric_limits<tscalar>::max() : 
                        history.begin()->first;
        }
}

#endif // NANOCV_SEARCH1D_H

