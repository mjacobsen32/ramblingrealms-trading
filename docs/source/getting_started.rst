===============
Getting Started
===============

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


^^^^^^^^^^^^^^^^^^
Polygon (optional)
^^^^^^^^^^^^^^^^^^
`Create a Polygon API token and save to local host <https://polygon.io/>`_

Not integrated yet.

^^^^^^^^^^^^
Setup wizard
^^^^^^^^^^^^
Use the setup wizard to configure api keys which the application will consume via a user cache object, stored at `~/.config/rr-trading/user_cache.json`

.. typer:: trading.cli.rr_trading:app.setup
   :prog: rr_trading setup
   :width: 80
   :show-nested:
   :make-sections:
   :theme: dimmed_monokai
