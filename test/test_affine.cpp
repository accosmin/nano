#include "loss.h"
#include "utest.h"
#include "trainer.h"
#include "criterion.h"
#include "math/epsilon.h"
#include "tasks/task_affine.h"
#include "layers/make_layers.h"

using namespace nano;

const auto idims = tensor_size_t(3);
const auto irows = tensor_size_t(10);
const auto icols = tensor_size_t(10);
const auto osize = tensor_size_t(10);
const auto count = tensor_size_t(1000);
const auto noise = epsilon2<scalar_t>();

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
        NANO_REQUIRE(*model == task);

        // create loss
        const auto losses =
        {
                nano::get_losses().get("square"),
                nano::get_losses().get("cauchy")
        };

        // create criteria
        const auto criterion = nano::get_criteria().get("avg");

        // create trainers
        const auto epochs = 1000;
        const auto policy = trainer_policy::stop_early;
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

                        // the average training loss value & error should be "small"
                        const auto opt_state = result.optimum_state();
                        NANO_CHECK_LESS(opt_state.m_train.m_value_avg, epsilon3<scalar_t>());
                        NANO_CHECK_LESS(opt_state.m_train.m_error_avg, epsilon3<scalar_t>());
                }
        }
}

NANO_END_MODULE()
