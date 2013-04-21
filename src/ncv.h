#ifndef NANOCV_H
#define NANOCV_H

#include "ncv_optimize.h"
#include "ncv_logger.h"
#include "ncv_random.h"
#include "ncv_timer.h"
#include "ncv_thread.h"
#include "ncv_image.h"
#include "ncv_stats.h"
#include "ncv_loss.h"

namespace ncv
{
        // initialize library (register objects, start worker pool ...)
        void init();
}

#endif // NANOCV_H
