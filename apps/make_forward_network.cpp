#include "task.h"
#include "model.h"
#include "task_util.h"
#include "text/cmdline.h"
#include "measure_and_log.h"
#include "layers/make_layers.h"
#include "layers/conv3d_params.h"

using namespace nano;

static conv3d_params_t conv3d_odims(const tensor3d_dims_t& idims, const tensor_size_t psize,
        const tensor_size_t min_ksize = 3, const tensor_size_t max_ksize = 7, const tensor_size_t omode = 4)
{
        for (auto ksize = max_ksize; ksize >= min_ksize; ksize -= 2)
        {
                for (auto omaps = iround(psize, omode); omaps > 0; omaps -= omode)
                {
                        const auto imaps = idims[0], irows = idims[1], icols = idims[2];
                        const auto krows = ksize, kcols = ksize;
                        const auto kconn = 1, kdrow = 1, kdcol = 1;

                        const auto params = conv3d_params_t{imaps, irows, icols, omaps, kconn, krows, kcols, kdrow, kdcol};
                        if (params.valid() && params.psize() < psize)
                        {
                                return params;
                        }
                }
        }

        return conv3d_params_t{};
}

static tensor3d_dims_t affine_odims(const tensor3d_dims_t& idims, const tensor_size_t psize,
        const tensor_size_t omode = 8)
{
        const auto osize = idiv(psize, 1 + nano::size(idims));

        return {iround(osize, omode), 1, 1};
}

int main(int argc, const char *argv[])
{
        // parse the command line
        cmdline_t cmdline("create the optimum forward network models");
        cmdline.add("", "task",                 "[" + concatenate(get_tasks().ids()) + "]");
        cmdline.add("", "task-params",          "task parameters (if any)", "-");
        cmdline.add("", "cdepth",               "number of convolution layers [0, 100]", "0");
        cmdline.add("", "mdepth",               "number of affine layers [0, 100]", "0");
        cmdline.add("", "psizeo",               "number of parameters for the first layer [1, 4M]", "128");
        cmdline.add("", "psizex",               "number of parameters for the other layers [1, 4M]", "128");
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
        const auto cmd_psizeo = clamp(cmdline.get<tensor_size_t>("psizeo"), 1, 4 * 1024 * 1024);
        const auto cmd_psizex = clamp(cmdline.get<tensor_size_t>("psizex"), 1, 4 * 1024 * 1024);
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
                const auto min_ksize = 3;
                const auto max_ksize = 3;//clamp(iround(std::min(idims[1], idims[2]) / 4, 2) + 1, min_ksize, 7);
                const auto psize = i == 0 ? cmd_psizeo : cmd_psizex;

                const auto params = conv3d_odims(idims, psize, min_ksize, max_ksize);
                if (params.valid())
                {
                        const auto omaps = params.omaps();
                        const auto krows = params.krows();
                        const auto kcols = params.kcols();
                        const auto kconn = params.kconn();
                        const auto kdrow = params.kdrow();
                        const auto kdcol = params.kdcol();

                        model_params += make_conv3d_layer(omaps, krows, kcols, kconn, cmd_activation, kdrow, kdcol);
                        idims = params.odims();
                }
                else
                {
                        break;
                }
        }

        // create model: add affine layers
        for (tensor_size_t i = 0; i < cmd_mdepth; ++ i)
        {
                const auto psize = i + cmd_cdepth == 0 ? cmd_psizeo : cmd_psizex;

                idims = affine_odims(idims, psize);
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
