#ifndef NANOCV_H
#define NANOCV_H

#include "ncv_optimize.h"
#include "ncv_geom.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include "ncv_timer.h"
#include "ncv_thread.h"
#include "ncv_image.h"
#include "ncv_stats.h"
#include "ncv_loss.h"
#include "ncv_task.h"
#include "ncv_model.h"

namespace ncv
{
        // current version
        static const size_t MAJOR_VERSION = 0;
        static const size_t MINOR_VERSION = 1;

        inline string_t version()
        {
                return text::to_string(MAJOR_VERSION) + "." +
                       text::to_string(MINOR_VERSION);
        }

        // initialize library (register objects, start worker pool ...)
        void init();
}

#endif // NANOCV_H
