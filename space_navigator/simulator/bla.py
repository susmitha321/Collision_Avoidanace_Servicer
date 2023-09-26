## created plot iteration file by me. 

def plot_iteration(self, epoch):

        r_coll_prob = self.reward_components["coll_prob"]
        r_fuel = self.reward_components["fuel"]
        r_traj_dev = sum(self.reward_components["traj_dev"])
        r_prob_dockpos = self.reward_components["dock_prob_relpos"]
        r_dockvel = self.reward_components["dock_relvel"]
        
        s = f"""Epoch: {epoch}\n
Collision Probability: {self.prob_arr[-1]:.5f}
Fuel Consumption: {self.fuel_cons_arr[-1]:.5f} (|dV|)
Trajectory Deviation:
    a: {self.traj_dev[0]:.5} (m);
    e: {self.traj_dev[1]:.5};
    i: {self.traj_dev[2]:.5} (rad);
    W: {self.traj_dev[3]:.5} (rad);
    w: {self.traj_dev[4]:.5} (rad);
    M: {self.traj_dev[5]:.5} (rad).
Docking probability position: {self.dock_pos_arr[-1]:.5}
Docking relvel: {self.dock_vel_arr[-1]:.5}
Reward Components:
    R Collision Probability: {r_coll_prob:.5};
    R Fuel Consumption: {r_fuel:.5};
    R Trajectory Deviation: {r_traj_dev:.5};
    R Docking in position: {r_prob_dockpos:.5};
    R Dcoking in velocity: {r_dockvel:.5}.
Total Reward: {self.reward_arr[-1]:.5}.
"""
        if self.curr_alert_info:
            s_alert = f"""Danger of collision!\n
Object:                   {self.curr_alert_info["debris_name"]};
Probability:              {self.curr_alert_info["probability"]};
Miss distance:            {self.curr_alert_info["distance"]};
Epoch:                    {pk.epoch(self.curr_alert_info["epoch"])};
Seconds before collision: {self.curr_alert_info["sec_before_collision"]}.
"""  
        else:
            s_alert = "No danger."
        self.subplot_3d.text2D(-0.3, 0.7, s,
                               transform=self.subplot_3d.transAxes)
        self.subplot_3d.text2D(0.35, 1.07, s_alert,
                               transform=self.subplot_3d.transAxes)