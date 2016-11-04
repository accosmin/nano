#include "loss.h"
#include "utest.h"
#include "trainer.h"
#include "criterion.h"
#include "tasks/task_affine.h"
#include "layers/make_layers.h"

using namespace nano;

const auto idims = tensor_size_t(3);
const auto irows = tensor_size_t(10);
const auto icols = tensor_size_t(10);
const auto osize = tensor_size_t(10);
const auto count = tensor_size_t(1000);
const auto noise = scalar_t(0.001);

NANO_BEGIN_MODULE(test_affine)

NANO_CASE(construction)
{
        affine_task_t task(to_params(
                "idims", idims, "irows", irows, "icols", icols, "osize", osize, "count", count, "noise", noise));

        task.load();

        NANO_CHECK_EQUAL(task.idims(), idims);
        NANO_CHECK_EQUAL(task.irows(), irows);
        NANO_CHECK_EQUAL(task.icols(), icols);
        NANO_CHECK_EQUAL(task.osize(), osize);
        NANO_CHECK_EQUAL(task.n_samples(), count);
}

NANO_CASE(training)
{
        affine_task_t task(to_params(
                "idims", idims, "irows", irows, "icols", icols, "osize", osize, "count", count, "noise", noise));

        task.load();

        // create model
        const auto model_config = make_output_layer(task.osize());
        const auto model = nano::get_models().get("forward-network", model_config);
        NANO_CHECK_EQUAL(model->resize(task, true), true);

        // create loss
        const auto losses =
        {
                nano::get_losses().get("square"),
                nano::get_losses().get("cauchy")
        };

        // create criteria
        const auto criterion = nano::get_criteria().get("avg");

        // create trainers
        const auto epochs = 100;
        const auto policy = trainer_policy::all_epochs;
        const auto trainers =
        {
                nano::get_trainers().get("batch", to_params("opt", "gd", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("batch", to_params("opt", "cgd", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("batch", to_params("opt", "lbfgs", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "sg", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "sgm", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "ngd", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "adam", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "adagrad", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "adadelta", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "ag", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "agfr", "epochs", epochs, "policy", policy)),
                nano::get_trainers().get("stoch", to_params("opt", "aggr", "epochs", epochs, "policy", policy))
        };

        // check training
        for (const auto& loss : losses)
        {
                for (const auto& trainer : trainers)
                {
                        model->random_params();

                        const auto fold = size_t(0);
                        const auto threads = size_t(1);
                        const auto result = trainer->train(task, fold, threads, *loss, *criterion, *model);

                        // todo
                }
        }
}

NANO_END_MODULE()
