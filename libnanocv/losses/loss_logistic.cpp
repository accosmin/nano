#include "loss_logistic.h"
#include "libnanocv/util/math.hpp"
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

                const vector_t edges = -targets.array() * scores.array();

                return (edges.array() < scalar_t(0)).count();
        }

        scalar_t logistic_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                const vector_t edges = -targets.array() * scores.array();

                return (edges.array().exp() + scalar_t(1)).log().sum();
        }
        
        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());
                
                const vector_t edges = -targets.array() * scores.array();

                return -targets.array() * edges.array().exp() / (edges.array().exp() + scalar_t(1));
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
