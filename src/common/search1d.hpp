#ifndef NANOCV_SEARCH1D_H
#define NANOCV_SEARCH1D_H

#include <cmath>
#include <map>

namespace ncv
{
        namespace detail 
        {
                template
                <
                        typename toperator,
                        typename tscalar,
                        typename tmap
                >
                typename tmap::mapped_type update_history(const toperator& op, tscalar param, tmap& history)
                {
                        typename tmap::iterator it = history.find(param);
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
        ///     using a greedy approach on the logarithmic scale in the range [minlog, maxlog].
        ///
        /// \returns the result associated with the optimum (log) paramerer.
        ///
        template
        <
                typename toperator,     ///< toperator(tscalar param) returns a tscalar score
                typename tscalar
        >
        auto min_search1d(const toperator& op, tscalar minlog, tscalar maxlog, tscalar epslog) -> decltype(op(tscalar(0)))
        {
                typedef decltype(op(tscalar(0))) tresult;
                typedef typename std::map<tscalar, tresult>::value_type tvalue;
                
                std::map<tscalar, tresult> history;
                
                tscalar bestlog = (maxlog + minlog) / 2;
                
                for (   tscalar distlog = (maxlog - minlog) / 2;
                        distlog > epslog && epslog > tscalar(0);
                        distlog /= 2)
                {
                        const tscalar param1 = bestlog - distlog;
                        const tscalar param2 = bestlog;
                        const tscalar param3 = bestlog + distlog;
                        
                        const tresult score1 = detail::update_history(op, param1, history);
                        const tresult score2 = detail::update_history(op, param2, history);
                        const tresult score3 = detail::update_history(op, param3, history);
                        
                        if (score1 < score2 && score1 < score3)
                        {
                                bestlog = param1;
                        }
                        else if (score2 < score3 && score2 < score1)
                        {
                                bestlog = param2;
                        }
                        else if (score3 < score1 && score3 < score2)
                        {
                                bestlog = param3;
                        }
                        else
                        {
                                break;
                        }
                }
                
                return  history.empty() ?
                        tresult() :
                        std::min_element(history.begin(), history.end(), [] (const tvalue& res1, const tvalue& res2)
                        {
                                return res1.second < res2.second;
                        })->second;
        }
}

#endif // NANOCV_SEARCH1D_H

