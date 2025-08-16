=========
Profiling
=========

**with py-spy`https://github.com/benfred/py-spy`_**
.. code-block:: console

    py-spy record - o ./profile.svg -- <commmand>

**speedscope integration`https://www.speedscope.app/`_**

.. code-block:: console

    py-spy record -f speedscope -o ./profile.svg -- <commmand>

*e.g.*

.. code-block:: console

    py-spy record -f speedscope -o ./profile.svg -- rr_trading alg train -c ./trading/configs/multi_ticker_ppo.json

.. image:: ../../assets/speedscope.png
    :alt: speedscope example
    :width: 600px