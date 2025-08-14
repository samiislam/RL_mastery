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

    avg_rewards = np.zeros(num_steps, dtype=float)
    optimal_action_pct = np.zeros(num_steps, dtype=float)

    for s in range(num_sets):
        q_star = np.random.normal(0.0, 1.0, size=num_arms)
        optimal_a = int(np.argmax(q_star))

        #Q = np.zeros(num_arms, dtype=float)
        Q = np.full(num_arms, av_init, dtype=float)
        N = np.zeros(num_arms, dtype=int)

        for t in range(num_steps):
            # epsilon-greedy with random tie-breaking
            if np.random.rand() < epsilon:
                a = np.random.randint(num_arms)
            else:
                max_q = np.max(Q)
                ties = np.flatnonzero(Q == max_q)
                a = int(np.random.choice(ties))

            # sample reward and update sample-average estimate immediately
            r = np.random.normal(q_star[a], 1.0)
            N[a] += 1
            Q[a] += (r - Q[a]) / N[a]

            # accumulate metrics
            avg_rewards[t] += r
            if a == optimal_a:
                optimal_action_pct[t] += 1

    # average over problems
    avg_rewards /= num_sets
    optimal_action_pct = 100.0 * optimal_action_pct / num_sets

    return (epsilon, avg_rewards, optimal_action_pct)