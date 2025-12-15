===========
Usage (CLI)
===========

--------
Training
--------

.. typer:: trading.cli.rr_trading:app.alg.train
   :prog: alg train
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai


.. note::

    tensorboard output will be saved to `<out_dir>/tensorboard` if configured.



-----------
Backtesting
-----------

.. typer:: trading.cli.rr_trading:app.alg.backtest
   :prog: alg backtest
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai

^^^^^^^^
Analysis
^^^^^^^^

.. typer:: trading.cli.rr_trading:app.alg.analysis
   :prog: alg analysis
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai

.. note::
    plots will be saved to `<out_dir>/backtest` and/or rendered in browser if configured

**Sample Backtest Results Rendering**

.. image:: ../../assets/backtest_plots.png
    :alt: speedscope example
    :width: 600px

**Stats rendering**
::

    Stats:
        Start                          2023-07-17 04:00:00+00:00
        End                            2025-06-04 04:00:00+00:00
        Period                                 474 days 00:00:00
        Start Value                                    1000000.0
        End Value                                     1181830.89
        Total Return [%]                               18.183089
        Benchmark Return [%]                            4.551781
        Max Gross Exposure [%]                         99.999854
        Total Fees Paid                                      0.0
        Max Drawdown [%]                                22.01912
        Max Drawdown Duration                  251 days 00:00:00
        Total Trades                                         222
        Total Closed Trades                                  221
        Total Open Trades                                      1
        Open Trade PnL                                1906.52112
        Win Rate [%]                                   57.466063
        Best Trade [%]                                 15.328848
        Worst Trade [%]                                -9.230098
        Avg Winning Trade [%]                           1.155003
        Avg Losing Trade [%]                           -1.281929
        Avg Winning Trade Duration    69 days 17:00:28.346456692
        Avg Losing Trade Duration     83 days 15:19:08.936170213
        Profit Factor                                   1.146855
        Expectancy                                    814.137416
        Sharpe Ratio                                    0.639243
        Calmar Ratio                                    0.623497
        Omega Ratio                                     1.161858
        Sortino Ratio                                   1.006545

-------------
Trading
-------------
Under construction


^^^^^^^^^^^^^
Paper Trading
^^^^^^^^^^^^^

There are 3 seperate clients for paper trading:
  1. Alpaca: Alpaca will interact with a paper portfolio via Alpaca API. The user does not have the ability to backtest any states prior to the current market state. This is recommended for true forward-testing.
  2. Local: Local paper trading simulates a live trading environment but will save all portfolio states to the local disk, {`open_positions`, `closed_positions`, `portfolio_history`, `account`}.
  3. Remote: Remote paper trading simulates a live trading environment but will save all portfolio states to a remote database, currently only supporting a S3 instances. (tested on an R2)

.. typer:: trading.cli.rr_trading:app.paper-trade
   :prog: paper-trade
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai

^^^^^^^^^^^^^
Live Trading
^^^^^^^^^^^^^

.. warning::
    Live trading will execute real trades with real capital via the configured brokerage API from Alpaca. Ensure that you have configured your API keys properly and understand the risks involved in live trading. It is recommended to test your strategies thoroughly in a paper trading environment before deploying them in live trading. The `paper-trade` api command is recommended for this purpose.

.. note::
    Ensure that your account has sufficient funds and that you are aware of the fees and commissions associated with live trading. Commission fees, slippage, and failed market orders can all impact the performance of your trading strategy; when training the agent these factors should be taken into account to better simulate live trading conditions.

.. typer:: trading.cli.rr_trading:app.live-trade
   :prog: live-trade
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai
