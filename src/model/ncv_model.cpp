#include "ncv_model.h"
#include "ncv_logger.h"

namespace ncv
{
        //-------------------------------------------------------------------------------------------------

        void model_t::test(const task_t& task, const fold_t& fold, const loss_t& loss,
                scalar_t& lvalue, scalar_t& lerror) const
        {
                lvalue = lerror = 0.0;

                size_t cnt = 0;

                const isamples_t& isamples = task.fold(fold);
                for (size_t s = 0; s < isamples.size(); s ++)
                {
                        const isample_t& isample = isamples[s];

                        const image_t& image = task.image(isample.m_index);
                        const vector_t target = image.get_target(isample.m_region);
                        if (image.has_target(target))
                        {
                                vector_t output;
                                process(image.get_input(isample.m_region), output);

                                lvalue += loss.value(target, output);
                                lerror += loss.error(target, output);
                                ++ cnt;
                        }
                }

                const scalar_t inv = (cnt == 0) ? 1.0 : 1.0 / cnt;
                lvalue *= inv;
                lerror *= inv;
        }

        //-------------------------------------------------------------------------------------------------

        bool model_t::train(const task_t& task, const fold_t& fold, const loss_t& loss)
        {
                if (fold.second != protocol::train)
                {
                        log_error() << "cannot only train models with training samples!";
                        return false;
                }

                m_rows = task.n_rows();
                m_cols = task.n_cols();
                m_outputs = task.n_outputs();

                resize();

                return _train(task, fold, loss);
        }

        //-------------------------------------------------------------------------------------------------

        void model_t::process(const image_t& image, coord_t x, coord_t y, vector_t& output) const
        {
                const vector_t input = image.get_input(geom::make_rect(x, y, n_cols(), n_rows()));
                return process(input, output);
        }

        //-------------------------------------------------------------------------------------------------
}
