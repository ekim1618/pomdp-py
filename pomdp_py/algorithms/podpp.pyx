from pomdp_py.framework.basics cimport Agent, State, Action, Observation,\
    MacroAction, sample_generative_model
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.representations.belief.particles cimport particle_reinvigoration
from pomdp_py.utils import typ

import time
import numpy as np
from tqdm import tqdm

np_generator = np.random.default_rng(seed=42)

cdef class TreeNode:
    def __init__(self):
        self.children = {}

    def __getitem__(self, key):
        return self.children.get(key, None)

    def __setitem__(self, key, value):
        self.children[key] = value

    def __contains__(self, key):
        return key in self.children

cdef class QNode(TreeNode):
    """An action node of the belief tree."""
    def __init__(self, num_visits=0, R=0.0, pref=0.0):
        self.num_visits = num_visits
        self.R = R
        self.pref = pref
        self.children = {}
    def __str__(self):
        return typ.red(f'QNode<N:{self.num_visits}, R:{self.R}, Pref:{self.pref} | {(self.children.keys())}>')

    def __repr__(self):
        return self.__str__()


cdef class VNode(TreeNode):
    """A belief node of the belief tree."""
    def __init__(self, num_visits=0, V=0.0):
        self.num_visits = num_visits
        self.V = V
        self.children = {}

    def __str__(self):
        return typ.green(f'VNode<N:{self.num_visits}, V:{self.V} | {(self.children.keys())}>')

    def __repr__(self):
        return self.__str__()

    def print_children_value(self):
        for action in self.children:
            print("   action %s: %.3f" % (str(action), self[action].value))

    cpdef argmax(VNode self):
        """argmax(VNode self)
        Returns the action of the child with highest preference"""
        cdef Action action, best_action
        cdef float best_pref = float("-inf")
        best_action = None
        for action in self.children:
            if self[action].pref > best_pref:
                best_action = action
                best_pref = self[action].pref
        return best_action

    @property
    def value(self):
        best_action = max(self.children, key=lambda action: self.children[action].pref)
        return self.children[best_action].pref
cdef class VNodeParticles(VNode):

    def __init__(self, num_visits=0, V=0, belief=Particles([])):
        self.num_visits = num_visits
        self.V = V
        self.belief = belief
        self.children = {}

    def __str__(self):
        return typ.green(f'VNode<N:{self.num_visits}, V:{self.V}, Num Particles:{len(self.belief)} | {(self.children.keys())}>')

    def __repr__(self):
        return self.__str__()
cdef class RootVNode(VNode):
    def __init__(self, num_visits, history):
        VNode.__init__(self, num_visits)
        self.history = history
    @classmethod
    def from_vnode(cls, vnode, history):
        """from_vnode(cls, vnode, history)"""
        rootnode = RootVNode(vnode.num_visits, history)
        rootnode.children = vnode.children
        return rootnode

cdef class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits, history, belief=Particles([])):
        # vnodeobj = VNodeParticles(num_visits, value, belief=belief)
        RootVNode.__init__(self, num_visits, history)
        self.belief = belief
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNodeParticles(vnode.num_visits, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode

# TODO: Add action prior?
cdef class PODPP(Planner):
    """ PODPP (Partially Observable DPP) is an extension of :cite 'azar2011dpp'
    to partially-observable domains.

    __init__(self,

    Args:
        max_depth (int): Depth of the DPP search tree. Default: 5.
        planning_time (float), amount of time given to each planning step (seconds). Default: -1.
            if negative, then planning terminates when number of simulations `num_sims` reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        num_sims (int): Number of simulations for each planning step. If negative,
            then will terminate when planning_time is reached.
            If both `num_sims` and `planning_time` are negative, then the planner will run for 1 second.
        rollout_policy (RolloutPolicy): rollout policy. Default: RandomRollout.
        action_prior (ActionPrior): a prior over preferred actions given state and history.
        show_progress (bool): True if print a progress bar for simulations.
        pbar_update_interval (int): The number of simulations to run after each update of the progress bar,
            Only useful if show_progress is True; You can set this parameter even if your stopping criteria
            is time.

        discount_factor (float): The discount factor. Default: 0.99.
        temperature (float): A positive real number indicating the temperature of the soft-max. Default 10.
        default_preference (float): The default preference value. Default 0.0.
        max_depth (int): Maximum depth of the search tree. Default: 30.
        rollout_depth (int): Maximum depth of the rollout. Default: 30.
        init_particle_count (int): Number of particles representing the initial belief. Default: 1000.
        reinvigoration_target (int): The target number of particles whenever reinvigoration occurs. Default: 100.
        next_belief_generation_attempts (int): The number of attempts at generating the next belief given a b, a, o triple. Default: 1000
        timeout (float): The amount of time given to each planning step (seconds). Default: 1.0
        episode_count (int): The episode budget at each planning step. Default: -1 (i.e. use timeout).
    """

    def __init__(self,
                 planning_time=-1., num_sims=-1,
                 max_depth=5, rollout_depth=10,
                 discount_factor=0.9,
                 temperature=0.1, macro_action_length=10,
                 num_visits_init=0, V_init=0., R_init=0., pref_init=0.,
                 rollout_policy=None,
                 action_prior=None, show_progress=False, pbar_update_interval=5):
        self._planning_time = planning_time
        self._num_sims = num_sims
        if self._num_sims < 0 and self._planning_time < 0:
            self._planning_time = 1.
        self._max_depth = max_depth
        self._rollout_depth = rollout_depth
        self._discount_factor = discount_factor
        self._temperature = temperature
        self._macro_action_length = macro_action_length

        self._num_visits_init = num_visits_init
        self._V_init = V_init
        self._R_init = R_init
        self._pref_init = pref_init

        self._rollout_policy = rollout_policy
        self._action_prior = action_prior

        self._show_progress = show_progress
        self._pbar_update_interval = pbar_update_interval

        # to simplify function calls; plan only for one agent at a time
        self._agent = None
        self._last_num_sims = -1
        self._last_planning_time = -1

    # TODO: Repair references.
    @property
    def __str__(self):
        return f'PO-DPP Params[' \
               f'exploration_const:{self._exploration_const}, ' \
               f'discount_factor:{self._discount_factor}, ' \
               f'temperature:{self._temperature}, ' \
               f'default_preference:{self._default_preference}, ' \
               f'max_depth:{self._max_depth}, ' \
               f'rollout_depth:{self._rollout_depth}, ' \
               f'init_particle_count:{self._init_particle_count}, ' \
               f'reinvigoration_target:{self._reinvigoration_target}, ' \
               f'next_belief_generation_attempts:{self._next_belief_generation_attempts}, ' \
               f'timeout:{self._timeout}, ' \
               f'episode_count:{self._episode_count}]'

    def __repr__(self):
        return self.__str__()
    @property
    def update_agent_belief(self):
        """True if planner's update function also updates agent's
        belief."""
        return True

    @property
    def last_num_sims(self):
        """Returns the number of simulations ran for the last `plan` call."""
        return self._last_num_sims

    @property
    def last_planning_time(self):
        """Returns the amount of time (seconds) ran for the last `plan` call."""
        return self._last_planning_time

    @property
    def max_depth(self):
        return self._max_depth

    @property
    def discount_factor(self):
        return self._discount_factor

    @property
    def action_prior(self):
        return self._action_prior

    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, sims_count, time_taken = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
        return action
    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=None):
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        
        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, Particles):
            raise TypeError("agent's belief is not represented in particles.\n"\
                            "PODPP not usable. Please convert it to particles.")

        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation],
                                                   agent.history)
        tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(tree_belief,
                                                 len(agent.init_belief.particles),
                                                 state_transform_func=state_transform_func))
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)

    def clear_agent(self):
        self._agent = None  # forget about current agent so that can plan for another agent.
        self._last_num_sims = -1

    cpdef _search(self):
        cdef int sims_count = 0
        cdef double start_time, time_taken
        pbar = self._initialize_progress_bar()
        start_time = time.time()

        while not self._should_stop(sims_count, start_time):
            state = self._agent.sample_belief()
            self._perform_simulation(state)
            sims_count += 1
            self._update_progress(pbar, sims_count, start_time)

        self._finalize_progress_bar(pbar)
        best_action = self._agent.tree.argmax() # TODO: Define (use sample_policy?)
        time_taken = time.time() - start_time
        return best_action, time_taken, sims_count

    cdef _initialize_progress_bar(self):
        if self._show_progress:
            total = self._num_sims if self._num_sims > 0 else self._planning_time
            return tqdm(total=total)
    cpdef _perform_simulation(self, state):
        self._simulate(state, self._agent.history, self._agent.tree, None, None, 0)

    cdef bint _should_stop(self, int sims_count, double start_time):
        cdef float time_taken = time.time() - start_time
        if self._num_sims > 0:
            return sims_count >= self._num_sims
        else:
            return time_taken > self._planning_time

    cdef _update_progress(self, pbar, int sims_count, double start_time):
        if self._show_progress:
            pbar.n = sims_count if self._num_sims > 0 else round(time.time() - start_time, 2)
            pbar.refresh()

    cdef _finalize_progress_bar(self, pbar):
        if self._show_progress:
            pbar.close()
    cpdef _simulate(self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth):
        if depth > self._max_depth or state.terminal:
            return self._rollout(state, history, root, depth)
        if root is None:
            if self._agent.tree is None:
                root = self._VNode(root=True)
                self._agent.tree = root
                if self._agent.tree.history != self._agent.history:
                    raise ValueError("Unable to plan for the given history.")
            else:
                root = self._VNode()

            if parent is not None:
                parent[observation] = root

        root.belief.add(state)
        cdef int nsteps
        cdef Action action
        action = self._agent._policy_model.sample(state)
        # action = self.sample_policy(root) # TODO: Seems justified by the theoretical results.
        next_state, observation, reward, nsteps = sample_generative_model(self._agent,
                                                                          state, action,
                                                                          self.discount_factor)

        # TODO: Remove at some point. For debugging in visualiser at the moment.
        # self._env.transition_model._pyb_env.set_config(next_s._position + (0, 0, 0, 0))

        # Create nodes if not created already.
        if root[action] is None:
            root[action] = QNode(num_visits=self._num_visits_init, R=self._R_init, pref=self._pref_init)

        # Update counts and estimates
        root.num_visits += 1
        root[action].num_visits += 1

        root[action].R = root[action].R + (reward - root[action].R) / root[action].num_visits

        root[action].pref += root[action].R + (self._discount_factor ** nsteps) \
                             * self._simulate(next_state,
                                          history + ((action, observation),),
                                          root[action][observation],
                                          root[action],
                                          observation,
                                          next_state,
                                          depth + nsteps) \
                             - root.V

        root.V = root.value

        return root.V
    cpdef _rollout(self, State state, tuple history, VNode root, int depth):
        cdef Action action
        cdef float discount = 1.0
        cdef float total_discounted_reward = 0
        cdef State next_state
        cdef Observation observation
        cdef float reward

        while depth < self._max_depth:
            action = self._rollout_policy.rollout(state, history)
            next_state, observation, reward, nsteps = sample_generative_model(self._agent, state, action, self._discount_factor)
            history = history + ((action, observation),)
            depth += nsteps
            total_discounted_reward += reward * discount
            discount *= (self._discount_factor**nsteps)
            state = next_state
        return total_discounted_reward
    # def sample_policy(self, root: BeliefNode, execute=False):
    #
    #     if root.N == 0:
    #         # print("Sampling policy from virgin root.")
    #         return self._agent._ref_policy_model.sample(self.sample_belief(root))
    #
    #     max_action = None
    #     max_pref = -np.inf
    #
    #     for a in root.children:
    #         if root[a].pref > max_pref:
    #             max_pref = root[a].pref
    #             max_action = a
    #
    #     if execute:
    #         return max_action
    #     else:
    #         if max_pref < self._default_preference:
    #             return self._agent._ref_policy_model.sample(self.sample_belief(root))
    #
    #         return max_action

    def _VNode(self, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        if root:
            return RootVNodeParticles(self._num_visits_init,
                             self._agent.history,
                             belief=copy.deepcopy(self._agent.belief))

        else:
            return VNodeParticles(self._num_visits_init,
                                  belief=Particles([]))