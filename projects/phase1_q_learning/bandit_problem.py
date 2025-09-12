import numpy as np

def process_bandit_problem(epsilon=0.0, num_sets=2000, num_steps=1000, av_init=0.0):

    np.random.seed(37)

    # Standard 10-armed bandit testbed (Sutton & Barto):
    # - 2000 independent bandit problems
    # - 1000 time steps per problem
    # - q*(a) ~ N(0, 1) fixed per problem; rewards R_t ~ N(q*(A_t), 1)
    # - epsilon-greedy with sample-average updates
    # - report average reward per step and % optimal action over time

    num_arms = 10

    total_rewards = np.zeros(num_steps, dtype=float)
    optimal_action_pct = np.zeros(num_steps, dtype=float)

    for _ in range(num_sets):
        # generate the true q-values for the bandit problem
        q_star = np.random.normal(0.0, 1.0, size=num_arms)
        # find the optimal action
        optimal_a = int(np.argmax(q_star))

        # initialize the Q-values and the number of times each action has been chosen
        Q = np.full(num_arms, av_init, dtype=float) # Note that the initial Q-values chosen also affect the results
        N = np.zeros(num_arms, dtype=int)

        for t in range(num_steps):
            # epsilon-greedy with random tie-breaking
            if np.random.rand() < epsilon:
                # randomly choose an action
                a = np.random.randint(num_arms)
            else:
                # randomly choose an action from the actions with the highest Q-value
                # (In the book (p. 27) it says we should pick the action with the highest 
                # Q-value but we can have multiple actions with the highest Q-value)
                max_q = np.max(Q)
                ties = np.flatnonzero(Q == max_q)
                a = int(np.random.choice(ties))

            # sample reward chosen from a normal distribution with mean q_star[a] and standard deviation 1.0
            r = np.random.normal(q_star[a], 1.0)
            # update the number of times each action has been chosen
            N[a] += 1
            # update the Q-value for the chosen action using the sample-average update rule (our learning method)
            Q[a] += (r - Q[a]) / N[a]

            # accumulate metrics
            total_rewards[t] += r
            if a == optimal_a:
                optimal_action_pct[t] += 1

    # average over problems
    avg_rewards = total_rewards / num_sets
    optimal_action_pct = 100.0 * optimal_action_pct / num_sets

    return (epsilon, avg_rewards, optimal_action_pct)


def process_bandit_problem_gradual_epsilon(epsilon=0.0, num_sets=2000, num_steps=1000, av_init=0.0):

    np.random.seed(37)

    # Standard 10-armed bandit testbed (Sutton & Barto):
    # - 2000 independent bandit problems
    # - 1000 time steps per problem
    # - q*(a) ~ N(0, 1) fixed per problem; rewards R_t ~ N(q*(A_t), 1)
    # - epsilon-greedy with sample-average updates
    # - report average reward per step and % optimal action over time

    num_arms = 10

    total_rewards = np.zeros(num_steps, dtype=float)
    optimal_action_pct = np.zeros(num_steps, dtype=float)

    gradual_epsilon = (epsilon == 'gradual')

    for _ in range(num_sets):
        # generate the true q-values for the bandit problem
        # Does this mean that my true values are non-stationary?
        # Or do they have to be non-stationary per step?
        # Yes - they have to be non-stationary per step as mentioned in the book (p. 33, Exercise 2.5)
        q_star = np.random.normal(0.0, 1.0, size=num_arms)
        # find the optimal action
        optimal_a = int(np.argmax(q_star))

        # initialize the Q-values and the number of times each action has been chosen
        Q = np.full(num_arms, av_init, dtype=float) # Note that the initial Q-values chosen also affect the results
        N = np.zeros(num_arms, dtype=int)

        for t in range(num_steps):
            if gradual_epsilon:
                if t < num_steps * 0.1:
                    epsilon = 0.1
                elif t < num_steps * 0.2:
                    epsilon = 0.09
                elif t < num_steps * 0.3:
                    epsilon = 0.08
                elif t < num_steps * 0.4:
                    epsilon = 0.07
                elif t < num_steps * 0.5:
                    epsilon = 0.06
                elif t < num_steps * 0.6:
                    epsilon = 0.05
                elif t < num_steps * 0.7:
                    epsilon = 0.04
                elif t < num_steps * 0.8:
                    epsilon = 0.03
                elif t < num_steps * 0.9:
                    epsilon = 0.02
                else:
                    epsilon = 0.01

            # epsilon-greedy with random tie-breaking
            if np.random.rand() < epsilon:
                # randomly choose an action
                a = np.random.randint(num_arms)
            else:
                # randomly choose an action from the actions with the highest Q-value
                # (In the book (p. 27) it says we should pick the action with the highest 
                # Q-value but we can have multiple actions with the highest Q-value)
                max_q = np.max(Q)
                ties = np.flatnonzero(Q == max_q)
                a = int(np.random.choice(ties))

            # sample reward chosen from a normal distribution with mean q_star[a] and standard deviation 1.0
            # The reward variance is 1.0
            r = np.random.normal(q_star[a], 1.0)
            # update the number of times each action has been chosen
            N[a] += 1
            # update the Q-value for the chosen action using the sample-average update rule (our learning method)
            Q[a] += (r - Q[a]) / N[a]

            # accumulate metrics
            total_rewards[t] += r
            if a == optimal_a:
                optimal_action_pct[t] += 1

    # average over problems
    avg_rewards = total_rewards / num_sets
    optimal_action_pct = 100.0 * optimal_action_pct / num_sets

    return ('gradual' if gradual_epsilon else epsilon, avg_rewards, optimal_action_pct)


def process_bandit_problem_nonstationary(epsilon=0.1, num_sets=2000, num_steps=1000, av_method='sample-average'):

    np.random.seed(37)

    # Standard 10-armed bandit testbed (Sutton & Barto):
    # - 2000 independent bandit problems
    # - 1000 time steps per problem
    # - q*(a) ~ N(0, 1) fixed per problem; rewards R_t ~ N(q*(A_t), 1)
    # - epsilon-greedy with sample-average updates
    # - report average reward per step and % optimal action over time

    num_arms = 10

    total_rewards = np.zeros(num_steps, dtype=float)
    optimal_action_pct = np.zeros(num_steps, dtype=float)

    for _ in range(num_sets):
        # generate the true q-values for the bandit problem
        q_star = np.random.normal(0.0, 1.0, size=num_arms)

        # initialize the Q-values and the number of times each action has been chosen
        Q = np.full(num_arms, 0.0, dtype=float) # Note that the initial Q-values chosen also affect the results
        N = np.zeros(num_arms, dtype=int)
        o_prime = 0

        for t in range(num_steps):
            # generate the true q-values for the bandit problem
            q_star += np.random.normal(0.0, 0.01)
            # find the optimal action
            optimal_a = int(np.argmax(q_star))
            
            # epsilon-greedy with random tie-breaking
            if np.random.rand() < epsilon:
                # randomly choose an action
                a = np.random.randint(num_arms)
            else:
                # randomly choose an action from the actions with the highest Q-value
                # (In the book (p. 27) it says we should pick the action with the highest 
                # Q-value but we can have multiple actions with the highest Q-value)
                max_q = np.max(Q)
                ties = np.flatnonzero(Q == max_q)
                a = int(np.random.choice(ties))

            # sample reward chosen from a normal distribution with mean q_star[a] and standard deviation 1.0
            r = np.random.normal(q_star[a], 1.0)
            # update the number of times each action has been chosen
            N[a] += 1
            if av_method=='sample-average':
                # update the Q-value for the chosen action using the sample-average update rule (our learning method)
                Q[a] += (r - Q[a]) / N[a]
            elif av_method=='constant-step-size': 
                Q[a] += (r - Q[a]) * 0.1
            else:
                # Unbiased constant-step-size trick (p. 35, Exercise 2.7)
                o_prime += 0.1 * (1 - o_prime)
                beta = 0.1 / o_prime
                Q[a] += (r - Q[a]) * beta

            # accumulate metrics
            total_rewards[t] += r
            if a == optimal_a:
                optimal_action_pct[t] += 1

    # average over problems
    avg_rewards = total_rewards / num_sets
    optimal_action_pct = 100.0 * optimal_action_pct / num_sets

    return (av_method, avg_rewards, optimal_action_pct)


def process_bandit_problem_action_values(epsilon=0.0, num_sets=2000, num_steps=1000, av_init=0.0):

    np.random.seed(37)

    # Standard 10-armed bandit testbed (Sutton & Barto):
    # - 2000 independent bandit problems
    # - 1000 time steps per problem
    # - q*(a) ~ N(0, 1) fixed per problem; rewards R_t ~ N(q*(A_t), 1)
    # - epsilon-greedy with sample-average updates
    # - report average reward per step and % optimal action over time

    num_arms = 10

    total_rewards = np.zeros(num_steps, dtype=float)
    optimal_action_pct = np.zeros(num_steps, dtype=float)

    for _ in range(num_sets):
        # generate the true q-values for the bandit problem
        q_star = np.random.normal(0.0, 1.0, size=num_arms)
        # find the optimal action
        optimal_a = int(np.argmax(q_star))

        # initialize the Q-values and the number of times each action has been chosen
        Q = np.full(num_arms, av_init, dtype=float) # Note that the initial Q-values chosen also affect the results
        N = np.zeros(num_arms, dtype=int)

        for t in range(num_steps):
            # epsilon-greedy with random tie-breaking
            if np.random.rand() < epsilon:
                # randomly choose an action
                a = np.random.randint(num_arms)
            else:
                # randomly choose an action from the actions with the highest Q-value
                # (In the book (p. 27) it says we should pick the action with the highest 
                # Q-value but we can have multiple actions with the highest Q-value)
                max_q = np.max(Q)
                ties = np.flatnonzero(Q == max_q)
                a = int(np.random.choice(ties))

            # sample reward chosen from a normal distribution with mean q_star[a] and standard deviation 1.0
            r = np.random.normal(q_star[a], 1.0)
            # update the number of times each action has been chosen
            N[a] += 1
            # update the Q-value for the chosen action using the constant step-size parameter
            Q[a] += (r - Q[a]) * 0.1

            # accumulate metrics
            total_rewards[t] += r
            if a == optimal_a:
                optimal_action_pct[t] += 1

    # average over problems
    avg_rewards = total_rewards / num_sets
    optimal_action_pct = 100.0 * optimal_action_pct / num_sets

    return (av_init, epsilon, avg_rewards, optimal_action_pct)

def process_bandit_problem_ucb(epsilon=0.0, num_sets=2000, num_steps=1000, av_init=0.0, ucb_c=0):

    np.random.seed(37)

    # Standard 10-armed bandit testbed (Sutton & Barto):
    # - 2000 independent bandit problems
    # - 1000 time steps per problem
    # - q*(a) ~ N(0, 1) fixed per problem; rewards R_t ~ N(q*(A_t), 1)
    # - epsilon-greedy with sample-average updates
    # - report average reward per step and % optimal action over time

    num_arms = 10

    total_rewards = np.zeros(num_steps, dtype=float)
    optimal_action_pct = np.zeros(num_steps, dtype=float)

    for _ in range(num_sets):
        # generate the true q-values for the bandit problem
        # Does this mean that my true values are non-stationary?
        # Or do they have to be non-stationary per step?
        # Yes - they have to be non-stationary per step as mentioned in the book (p. 33, Exercise 2.5)
        q_star = np.random.normal(0.0, 1.0, size=num_arms)
        # find the optimal action
        optimal_a = int(np.argmax(q_star))

        # initialize the Q-values and the number of times each action has been chosen
        Q = np.full(num_arms, av_init, dtype=float) # Note that the initial Q-values chosen also affect the results
        N = np.zeros(num_arms, dtype=int)

        for t in range(num_steps):

            # epsilon-greedy with random tie-breaking
            if np.random.rand() < epsilon:
                # randomly choose an action
                a = np.random.randint(num_arms)
            else:
                # UCB - upper confidence bound
                if ucb_c > 0:
                    # Calculate UCB values for all actions
                    # If N(a) = 0, then a is considered to be a maximizing action
                    ucb_values = np.zeros(num_arms)
                    for i in range(num_arms):
                        if N[i] == 0:
                            # Untried actions are considered maximizing
                            ucb_values[i] = float('inf')
                        else:
                            # Standard UCB formula: Q(a) + c * sqrt(ln(t) / N(a))
                            ucb_values[i] = Q[i] + ucb_c * np.sqrt(np.log(t + 1) / N[i])
                    
                    # Handle ties by randomly choosing among actions with highest UCB value
                    max_ucb = np.max(ucb_values)
                    ties = np.flatnonzero(ucb_values == max_ucb)
                    a = int(np.random.choice(ties))
                    
                else:
                    # randomly choose an action from the actions with the highest Q-value
                    # (In the book (p. 27) it says we should pick the action with the highest 
                    # Q-value but we can have multiple actions with the highest Q-value)
                    max_q = np.max(Q)
                    ties = np.flatnonzero(Q == max_q)
                    a = int(np.random.choice(ties))
               

            # sample reward chosen from a normal distribution with mean q_star[a] and standard deviation 1.0
            # The reward variance is 1.0
            r = np.random.normal(q_star[a], 1.0)
            # update the number of times each action has been chosen
            N[a] += 1
            # update the Q-value for the chosen action using the sample-average update rule (our learning method)
            Q[a] += (r - Q[a]) / N[a]

            # accumulate metrics
            total_rewards[t] += r
            if a == optimal_a:
                optimal_action_pct[t] += 1

    # average over problems
    avg_rewards = total_rewards / num_sets
    optimal_action_pct = 100.0 * optimal_action_pct / num_sets

    return (f'UCB c={ucb_c}' if ucb_c > 0 else f'ε={epsilon}', avg_rewards, optimal_action_pct)