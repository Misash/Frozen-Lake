### MDP Value Iteration and Policy Iteration
import argparse
import numpy as np
import gymnasium as gym
import time
from lake_envs import *



np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(
    description="A program to run assignment 1 implementations.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--env",
    type=str,
    help="The name of the environment to run your algorithm on.",
    choices=["Deterministic-4x4-FrozenLake-v0", "Stochastic-4x4-FrozenLake-v0"],
    default="Deterministic-4x4-FrozenLake-v0", #change environment
)

parser.add_argument(
    "--render-mode",
    "-r",
    type=str,
    help="The render mode for the environment. 'human' opens a window to render. 'ansi' does not render anything.",
    choices=["human", "ansi"],
    default="human",
)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary of a nested lists
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""


def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    while True:
        delta = 0

        for s in range(nS):
            # Store the current value of state `s`
            v = value_function[s]

            # Initialize the total expected return to 0
            total_expected_return = 0

            # For each possible outcome when taking action according to the policy in state `s`
            for prob, next_state, rew, _ in P[s][policy[s]]:
                # Calculate the expected return for this outcome
                expected_return = prob * (rew + gamma * value_function[next_state])
                
                # Add the expected return to the total expected return
                total_expected_return += expected_return

            # Update the value of state `s` in the value function
            value_function[s] = total_expected_return

            # Calculate the absolute difference between the old and new value of state `s`
            difference = abs(v - value_function[s])

            # Update `delta` with the maximum difference
            delta = max(delta, difference)

        if delta < tol:
            break
    ############################
    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

    new_policy = np.zeros(nS, dtype="int")

    ############################
    # YOUR IMPLEMENTATION HERE #
    for state in range(nS):
        # Initialize an array to hold the Q-values for each action
        q_values = np.zeros(nA)

        # For each action in the action space
        for action in range(nA):
            # Initialize the total expected return to 0
            total_expected_return = 0

            # For each possible outcome when taking this action in this state
            for prob, next_state, rew, _ in P[state][action]:
                # Calculate the expected return for this outcome
                expected_return = prob * (rew + gamma * value_from_policy[next_state])
                
                # Add the expected return to the total expected return
                total_expected_return += expected_return

            # Update the Q-value for this action
            q_values[action] = total_expected_return

        # Update the policy for this state to be the action with the highest Q-value
        new_policy[state] = np.argmax(q_values)

    ############################
    return new_policy

def policy_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    iteration_count = 0  # Contador de iteraciones

    while True:
        iteration_count += 1  # Incrementar el contador en cada iteración
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        new_policy = policy_improvement(P, nS, nA, value_function, policy, gamma)

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy

    print(f"Policy Iteration completed in {iteration_count} iterations")
    return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    iteration_count = 0  # Contador de iteraciones

    while True:
        iteration_count += 1  # Incrementar el contador en cada iteración
        delta = 0

        for state in range(nS):
            v = value_function[state]
            q_values = np.zeros(nA)

            for action in range(nA):
                total_expected_return = 0
                for prob, next_state, rew, _ in P[state][action]:
                    expected_return = prob * (rew + gamma * value_function[next_state])
                    total_expected_return += expected_return

                q_values[action] = total_expected_return

            value_function[state] = max(q_values)
            difference = abs(v - value_function[state])
            delta = max(delta, difference)

        if delta < tol:
            break

    policy = policy_improvement(P, nS, nA, value_function, np.zeros(nS, dtype=int), gamma)
    print(f"Value Iteration completed in {iteration_count} iterations")
    return value_function, policy


def render_single(env, policy, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

    episode_reward = 0
    ob, _ = env.reset()
    for t in range(max_steps):
        env.render()
        time.sleep(0.25)
        a = policy[ob]
        ob, rew, done, _, _ = env.step(a)
        episode_reward += rew
        if done:
            break
    env.render()
    if not done:
        print(
            "The agent didn't reach a terminal state in {} steps.".format(
                max_steps
            )
        )
    else:
        print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":



    # read in script argument
    args = parser.parse_args()

    # Make gym environment
    env = gym.make(args.env, render_mode=args.render_mode)

    env.nS = env.nrow * env.ncol
    env.nA = 4

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_pi, 100)

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
    render_single(env, p_vi, 100)

