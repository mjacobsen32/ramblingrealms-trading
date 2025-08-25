===============
Getting Started
===============

This is the content of my new page.

.. _section-label:

Installation
-------------

^^^^^^
Poetry
^^^^^^

`Install Poetry <https://python-poetry.org/docs/#installation>`_


^^^^^^^^^^^^^^^
Project Install
^^^^^^^^^^^^^^^

.. code-block:: console

    poetry install


Setup
-----

^^^^^^^^^^^^^^^^^
Alpaca (required)
^^^^^^^^^^^^^^^^^
`Create an Alpaca API token and save to local host <https://alpaca.markets/>`_

The alpaca api is used for collecting market data and executing trades with a paper trading and live trading account.


^^^^^^^^^^^^^^^^^
Etrade (optional)
^^^^^^^^^^^^^^^^^
`Create an Etrade API token and save to local host <https://www.etrade.com/>`_

The etrade api is outdated and likely to be deprecated in the future. Until long term support picks up, Alpaca is the recommended option for live/paper trading.


^^^^^^^^^^^^^^^^^^
Polygon (optional)
^^^^^^^^^^^^^^^^^^
`Create a Polygon API token and save to local host <https://polygon.io/>`_

Not integrated yet.

^^^^^^^^^^^^
Setup wizard
^^^^^^^^^^^^

.. typer:: trading.cli.rr_trading:app.setup
   :prog: rr_trading setup
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai