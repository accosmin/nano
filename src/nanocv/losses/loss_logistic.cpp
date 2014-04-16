#include "loss_logistic.h"
#include <cassert>

namespace ncv
{
        /////////////////////////////////////////////////////////////////////////////////////////

        // to upper bound the {0, 1} loss
        static const scalar_t delta = std::exp(1.0) - 1.0;

        /////////////////////////////////////////////////////////////////////////////////////////

        static void _buffer(const vector_t& targets, const vector_t& scores,
                vector_t& zs, vector_t& ys, scalar_t& sumw, scalar_t& sumwy)
        {
                zs.resize(targets.rows());
                ys.resize(targets.rows());

                sumw = 0.0;
                sumwy = 0.0;

                // soft-max multi-class logistic loss
                for (auto o = 0; o < targets.rows(); o ++)
                {
                        const scalar_t z = std::exp(- scores[o] * targets[o]);
                        const scalar_t w = delta + z;           // weight
                        const scalar_t y = std::log(w);         // one-class logistic

                        zs[o] = z;
                        ys[o] = y;

                        sumw += w;
                        sumwy += w * y;
                }
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        logistic_loss_t::logistic_loss_t()
                :       loss_t(string_t(), "logistic loss")
        {
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        scalar_t logistic_loss_t::value(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                vector_t zs, ys;
                scalar_t sumw, sumwy;
                _buffer(targets, scores, zs, ys, sumw, sumwy);

                return sumwy / sumw;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
        
        vector_t logistic_loss_t::vgrad(const vector_t& targets, const vector_t& scores) const
        {
                assert(targets.size() == scores.size());

                vector_t zs, ys;
                scalar_t sumw, sumwy;
                _buffer(targets, scores, zs, ys, sumw, sumwy);

                const scalar_t isumw2 = 1.0 / (sumw * sumw);

                vector_t grads(targets.rows());
                for (auto o = 0; o < targets.rows(); o ++)
                {
                        grads[o] = targets[o] * zs[o] * (sumwy - sumw * ys[o] - sumw) * isumw2;
                }

                return grads;
        }

        /////////////////////////////////////////////////////////////////////////////////////////
}
