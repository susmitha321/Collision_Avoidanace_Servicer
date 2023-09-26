import numpy as np

num_values = 10
coll_prob_thr = 0.0001
fuel_thr = 15
traj_dev_thr=(100, 0.01, 0.01, 0.01, 0.01, None)
dock_pos_thr = 1
dock_vel_thr = 0.1
thr = np.concatenate(
            ([coll_prob_thr], [fuel_thr], traj_dev_thr, [dock_pos_thr], [dock_vel_thr]
        )).astype(np.float)

# Define the ranges of values for each threshold
coll_prob_range = np.linspace(0, 0.001, num_values)  # Adjust the range as needed
fuel_range = np.linspace(0, 30, num_values)  # Adjust the range as needed


element_ranges = [
    np.linspace(0, 200, num_values),   # Element 1
    np.linspace(0, 0.1, num_values),   # Element 2
    np.linspace(0, 0.1, num_values),   # Element 3
    np.linspace(0, 0.1, num_values),   # Element 4
    np.linspace(0, 0.1, num_values),   # Element 5
    np.linspace(0, 0.1, num_values)    # Element 6
]

# Create the traj_dev_range array by stacking the element ranges horizontally
traj_dev_range = np.column_stack(element_ranges)
dock_pos_range = np.linspace(0, 1, num_values)  # Adjust the range as needed

dock_vel_range = np.linspace(0, 0.5, num_values)  # Adjust the range as needed



def reward_func_0(value, thr, r_thr=-1,
                  thr_times_exceeded=2, r_thr_times_exceeded=-10):
    """Reward function.

    Piecewise linear function with increased penalty for exceeding the threshold.

    Args:
        value (float): value of the rewarded parameter.
        thr (float): threshold of the rewarded parameter. 
        r_thr (float): reward in case the parameter value is equal to the threshold.
        thr_times_exceeded (float): how many times should the parameter value exceed
            the threshold to be rewarded as r_thr_times_exceeded.
        r_thr_times_exceeded (float): reward in case the parameter value exceeds
            the threshold thr_times_exceeded times.

    Returns:
        reward (float): result reward.
    """
    if value <= thr:
        reward = value * r_thr / thr
    else:
        reward = (
            (-r_thr_times_exceeded + r_thr)
            * (1 - value / thr) / (thr_times_exceeded - 1)
            + r_thr
        )
    return reward

    ###
    # add very high reward if its docked but its not necessary.
    ###

def reward_func(values, thr, reward_func=reward_func_0, *args, **kwargs):
    """Returns reward values for np.array input.

    Args:
        values (np.array): array of values of the rewarded parameters.
        thr (np.array): array of thresholds of the rewarded parameter.
            if the threshold is np.nan, then the reward is 0
            (there is no penalty for the parameter).
        dangerous_debris: if there are any dangerous debris in conjunction
        reward_func (function): reward function.
        *args, **kwargs: additional arguments of reward_func.

    Returns: 
        reward (np.array): reward array.
    """

    def reward_thr(values, thr):
        return reward_func(values, thr, *args, **kwargs)
    reward_thr_v = np.vectorize(reward_thr)

    reward = np.zeros_like(values)
    id_nan = np.isnan(thr)
    id_not_nan = np.logical_not(id_nan)

    reward[id_nan] = 0.
      
    if np.count_nonzero(id_not_nan) != 0:
        reward[id_not_nan] = reward_thr_v(values[id_not_nan], thr[id_not_nan])
        
    # Handling the last two values differently
    
    dock_prob_relpos = values[-2]
    dock_relvel = values[-1]
    
    # Reward logic for docking position
    if dock_prob_relpos == 1:  # If docked
        reward[-2] = 0
    else:  # If not docked
        reward[-2] = -1000
        
    # Reward logic for docking relative velocity
    if dock_relvel < thr[-1]:  # If velocity is below threshold
        reward[-1] = 0
    else:  # If velocity is above threshold
        reward[-1] = -1000

    return reward


# Initialize lists to store results
individual_rewards_list = []
total_rewards_list = []
coll_prob_r_list = []
fuel_r_list = []
dock_prob_relpos_r_list = []
dock_relvel_r_list = []
for i in range(num_values):
    coll_prob_value = coll_prob_range[i]
    fuel_value = fuel_range[i]
    traj_dev_values = traj_dev_range[i]
    dock_pos_value = dock_pos_range[i]
    dock_vel_value = dock_vel_range[i]

    # Create a parameter set
    param_values = np.array([coll_prob_value, fuel_value] + list(traj_dev_values) + [dock_pos_value, dock_vel_value])

    # Compute rewards using the reward function
    dangerous_debris = False  # Set to True if necessary
    rewards = reward_func(param_values, thr)

    # Separate individual rewards for plotting
    coll_prob_r_list.append(rewards[0])
    fuel_r_list.append(rewards[1])
    dock_prob_relpos_r_list.append(rewards[8])
    dock_relvel_r_list.append(rewards[9])

    # Compute and store total reward
    total_rewards = np.sum(rewards)
    total_rewards_list.append(total_rewards)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(fuel_range, fuel_r_list)
plt.xlabel('Fuel Range')
plt.ylabel('Fuel Reward')
plt.title('Fuel Reward vs. Fuel Range')
plt.grid(True)

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(coll_prob_range, coll_prob_r_list)
plt.xlabel('coll')
plt.ylabel('coll Reward')
plt.title('coll reward vs coll range')
plt.grid(True)

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dock_pos_range, dock_prob_relpos_r_list)
plt.xlabel('pos Range')
plt.ylabel('pos Reward')
plt.title('pos Reward vs. pos Range')
plt.grid(True)

plt.show()

plt.figure(figsize=(10, 6))
plt.plot(dock_vel_range, dock_relvel_r_list)
plt.xlabel('vel Range')
plt.ylabel('vel Reward')
plt.title('vel Reward vs. vel Range')
plt.grid(True)

plt.show()
