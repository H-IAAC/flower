from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl


def main() -> None:
    # Configure the aggregation strategy
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 0.1.
    fraction_eval : float, optional
        Fraction of clients used during validation. Defaults to 0.1.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_eval_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    """

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1,
        fraction_eval=1,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        eval_fn=None,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=None,
    )

    # Start Flower server for 10 rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": 5}, strategy=strategy)


def fit_config(rnd: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 1, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 1,
        # "local_epochs": 1 if rnd < 2 else 2,
        "local_epochs": 3,
    }
    return config

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    config = {
        "batch_size": 32,
        #"eval_steps": 3 if rnd < 4 else 10,
        "eval_steps": 10,
    }
    return config

if __name__ == "__main__":
    main()
