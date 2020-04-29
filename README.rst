FinGym
**********

**Fingym is a toolkit for developing reinforcement learning algorithms tailored specifically for stock market trading.**  This is the ``fingym`` open-source library, which gives you access to a standardized set of environments.

`See What's New section below <#what-s-new>`_

``fingym`` makes no assumptions about the structure of your agent, and is compatible with any numerical computation library, such as TensorFlow or Theano. You can use it from Python code.

If you're not sure where to start, we recommend beginning with the
`docs <https://entrpn.github.io/fingym/>`_ on our site.

Basics
======

There are two basic concepts in reinforcement learning: the
environment (namely, the outside world) and the agent (namely, the
algorithm you are writing). The agent sends `actions` to the
environment, and the environment replies with `observations` and
`rewards` (that is, a score).

The core `fingym` interface is `Env <https://github.com/entrpn/fingym/blob/master/gym/envs/env.py>`_, which is
the unified environment interface. There is no interface for agents;
that part is left to you. The following are the ``Env`` methods you
should know:

- `reset(self)`: Reset the environment's state. Returns `observation`.
- `step(self, action)`: Step the environment by one timestep. Returns `observation`, `reward`, `done`, `info`.

Supported systems
-----------------

We currently support Python 3.5 -- 3.7. 

Installation
============

You can perform a minimal install of ``fingym`` with:

.. code:: shell

    git clone git clone https://github.com/entrpn/fingym
    cd fingym
    pipenv shell
    pipenv install -e .

If you prefer, you can do a minimal install of the packaged version directly from PyPI:

.. code:: shell

    pip install fingym
    
Environments
============

See the `fingym site <https://entrpn.github.io/fingym/#environments>`_.

Observations
============

See the `fingym site <https://entrpn.github.io/fingym/#observations>`_.

Actions
=======

See the `fingym site <https://entrpn.github.io/fingym/#spaces>`_.

Examples
========

See the ``examples`` directory.

- Run `examples/agents/buy_and_hold_agent.py <https://github.com/entrpn/fingym/blob/master/gym/examples/agents/buy_and_hold_agent.py>`_ to run a simple buy and hold agent.
- Run `examples/agents/random_agent.py <https://github.com/entrpn/fingym/blob/master/gym/examples/agents/random_agent.py>`_ to run a simple random agent.
- Run `examples/agents/dqn_agent.py <https://github.com/entrpn/fingym/blob/master/gym/examples/agents/dqn_agent.py>`_ to run a dqn agent.
- Run `examples/agents/evolutionary_agent.py <https://github.com/entrpn/fingym/blob/master/gym/examples/agents/evolutionary_agent.py>`_ to run a generic algorithm.
- Run `examples/agents/evolutionary_agent_w_crossover.py <https://github.com/entrpn/fingym/blob/master/gym/examples/agents/evolutionary_agent_w_crossover.py>`_ to run a generic algorithm using crossover.

Testing
=======

If you cloned this repo, add fingym to python path:

>> export PYTHONPATH=$PYTHONPATH:/path/to/fingym/fingym

We are using `pytest <http://doc.pytest.org>`_ for tests. You can run them via:

.. code:: shell

    pytest


.. _See What's New section below:

What's new
==========
- 2020-02-05: First release. 3 year spy intraday minute steps. 10 year daily steps.
- 2020-02-26: More environments from different symbols.
- 2020-04-14: Renamed package from `gym` to `fingym`