#include "loss_logistic.h"
#include "common/math.hpp"
#include <cassert>

namespace ncv
{
        logistic_loss_t::logistic_loss_t(const string_t& configuration)
                :       loss_t(configuration)
        {
        }

        scalar_t logistic_loss_t::error(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                size_t errors = 0;
                for (auto i = 0; i < scores.size(); i ++)
                {
                        const scalar_t edge = targets(i) * scores(i);
                        if (edge <= 0.0)
                        {
                                errors ++;
                        }
                }
                
                return errors;
        }

        scalar_t logistic_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                const vector_t edges = (- targets.array() * scores.array()).exp();
                
                return std::log1p(edges.sum());
        }
        
        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                const vector_t edges = (- targets.array() * scores.array()).exp();
                
                return - targets.array() * edges.array() / (1.0 + edges.sum());
        }

        indices_t logistic_loss_t::labels(const vector_t& scores) const
        {
                indices_t ret;
                for (auto i = 0; i < scores.size(); i ++)
                {
                        if (scores(i) > 0.0)
                        {
                                ret.push_back(i);
                        }
                }

                return ret;
        }
}
