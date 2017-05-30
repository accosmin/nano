#include "task.h"
#include "model.h"
#include "task_util.h"
#include "text/cmdline.h"
#include "measure_and_log.h"
#include "layers/make_layers.h"
#include "layers/conv3d_params.h"

using namespace nano;

static conv3d_params_t conv3d_odims(const tensor3d_dims_t& idims, const tensor_size_t psize)
{
        const auto imaps = idims[0], irows = idims[1], icols = idims[2];
        const auto kconn = 1, kdrow = 1, kdcol = 1;

        const auto min_ksize = 1;
        const auto max_ksize = 5;

        auto params = conv3d_params_t{imaps, irows, icols, psize, 1, 3, 3, 1, 1};
        for (auto ksize = max_ksize; ksize >= min_ksize && (!params.valid() || params.psize() > psize); ksize -= 2)
        {
                for (auto omaps = psize; omaps > 0 && params.valid() && params.psize() > psize; -- omaps)
                {
                        const auto krows = ksize;
                        const auto kcols = ksize;
                        params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};
                }
        }

        return params;
}

static tensor3d_dims_t affine_odims(const tensor3d_dims_t& idims, const tensor_size_t psize)
{
        const auto osize = idiv(psize, 1 + nano::size(idims));
        return {osize, 1, 1};
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("create the optimum forward network models");
        cmdline.add("", "task",                 "[" + concatenate(get_tasks().ids()) + "]");
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
        cmdline.add("", "cdepth",               "number of convolution layers [0, 100]", "0");
        cmdline.add("", "mdepth",               "number of affine layers [0, 100]", "0");
        cmdline.add("", "psize",                "number of parameters per layer [1, 4M]", "128");
        cmdline.add("", "activation",           "activation layer", "act-snorm");

        cmdline.process(argc, argv);

        if (!cmdline.has("task"))
        {
                cmdline.usage();
        }

        // check arguments and options
        const auto cmd_task = cmdline.get<string_t>("task");
        const auto cmd_task_params = cmdline.get<string_t>("task-params");
        const auto cmd_cdepth = clamp(cmdline.get<tensor_size_t>("cdepth"), 0, 100);
        const auto cmd_mdepth = clamp(cmdline.get<tensor_size_t>("mdepth"), 0, 100);
        const auto cmd_psize = clamp(cmdline.get<tensor_size_t>("psize"), 1, 4 * 1024 * 1024);
        const auto cmd_activation = cmdline.get("activation");

        // create & load task
        const auto task = get_tasks().get(cmd_task, cmd_task_params);

        measure_critical_and_log(
                [&] () { return task->load(); },
                "load task <" + cmd_task + ">");

        describe(*task, cmd_task);

        // create model: add convolution layers
        string_t model_params;
        tensor3d_dims_t idims = task->idims();
        for (tensor_size_t i = 0; i < cmd_cdepth; ++ i)
        {
                const auto params = conv3d_odims(idims, cmd_psize);
                model_params += make_conv3d_layer(params.omaps(), params.krows(), params.kcols(), params.kconn(),
                        cmd_activation, params.kdrow(), params.kdcol());
                idims = params.odims();
        }

        // create model: add affine layers
        for (tensor_size_t i = 0; i < cmd_mdepth; ++ i)
        {
                idims = affine_odims(idims, cmd_psize);
                model_params += make_affine_layer(nano::size(idims), cmd_activation);
        }

        // create model: add output layer
        model_params += make_output_layer(task->odims());

        // create feed-forward network
        const auto model = get_models().get("forward-network", model_params);
        model->configure(*task);
        model->random();
        model->describe();

        // OK
        return EXIT_SUCCESS;
}
