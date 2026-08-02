"""Dynamic pricing for hotel revenue management.

Modules
-------
``config``       structural constants shared by every other module
``demand``       estimation of demand primitives from historical bookings
``environment``  the pricing MDP simulator
``mdp``          explicit transition kernel and exact Value Iteration
``agents``       n-step Double Q-learning
``lp``           mixed-integer rate-schedule optimisation
``policies``     baseline and wrapped policies
``metrics``      paired Monte-Carlo evaluation and significance testing
``plotting``     shared figure style and export
"""
