#include "ncv.h"
#include "core/math/clamp.hpp"
#include "core/timer.h"
#include "core/logger.h"
#include "models/forward_network.h"
#include <boost/program_options.hpp>

int main(int argc, char *argv[])
{
        ncv::init();

        using namespace ncv;

        strings_t cmd_networks =
        {
                "",
                "conv8x8:convs=16;snorm",
                "conv8x8:convs=16;tanh",
                "conv8x8:convs=16;unit",
                "conv8x8:convs=16;snorm;conv8x8:convs=16;snorm",
                "conv8x8:convs=16;tanh;conv8x8:convs=16;tanh",
                "conv8x8:convs=16;unit;conv8x8:convs=16;unit"
        };

        const color_mode cmd_color = color_mode::rgba;
        const size_t cmd_rows = 32;
        const size_t cmd_cols = 32;
        const size_t cmd_outputs = 10;
        const size_t cmd_samples = 10000;

        for (size_t n = 0; n < cmd_networks.size(); n ++)
        {
                const string_t cmd_network = cmd_networks[n];

                log_info() << "<<< running network [" << cmd_networks[n] << "] ...";

                // create feed-forward network
                ncv::forward_network_t network(cmd_network);
                network.resize(cmd_rows, cmd_cols, cmd_outputs, cmd_color);

                // create random samples
                tensor3ds_t samples(cmd_samples, tensor3d_t(cmd_color == color_mode::luma ? 1 : 3, cmd_rows, cmd_cols));
                for (tensor3d_t& sample : samples)
                {
                        sample.random(-1.0, +1.0);
                }

                // process the samples
                ncv::timer_t timer;

                size_t count = 0;
                for (tensor3d_t& sample : samples)
                {
                        const vector_t output = network.value(sample);
                        count += output.size();
                }

                log_info() << "<<< processed [" << (count / cmd_outputs) << "] samples in " << timer.elapsed() << ".";
        }

        // OK
        log_info() << done;
        return EXIT_SUCCESS;
}
