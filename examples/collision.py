# Collision runs simulator with two objects, having collision
# at time 6600.

import argparse
import sys

import numpy as np
import pykep as pk

from space_navigator.api import Environment
from space_navigator.simulator import Simulator
from space_navigator.agent import TableAgent as Agent
from space_navigator.utils import read_environment

PROPAGATION_STEP = 0.000001


def main(args):
    parser = argparse.ArgumentParser()

    # simulator initialization
    parser.add_argument("-at_path", "--action_table_path", type=str,
                        default=None, required=False)
    parser.add_argument("-env", "--environment", type=str,
                        default="data/environments/collision.env", required=False)
    parser.add_argument("-s", "--step", type=float,
                        default=PROPAGATION_STEP, required=False)

    # simulator run args
    parser.add_argument("-v", "--visualize", type=str,
                        default="True", required=False)
    parser.add_argument("-n_v", "--n_steps_vis", type=int,
                        default=1000, required=False)
    parser.add_argument("-log", "--logging", type=str,
                        default="True", required=False)
    parser.add_argument("-e_s_prop", "--each_step_propagation", type=str,
                        default="False", required=False)
    parser.add_argument("-print", "--print_out", type=str,
                        default="True", required=False)

    args = parser.parse_args(args)

    # simulator initialization
    action_table_path = args.action_table_path
    env_path = args.environment
    step = args.step

    # simulator run args
    visualize = args.visualize.lower() == "true"
    n_steps_vis = args.n_steps_vis
    log = args.logging.lower() == "true"
    each_step_propagation = args.each_step_propagation.lower() == "true"
    print_out = args.print_out.lower() == "true"

    # simulation
    env = read_environment(env_path)
    if action_table_path:
        action_table = np.loadtxt(action_table_path, delimiter=',')
        agent = Agent(action_table)
    else:
        agent = Agent()

    simulator = Simulator(
        agent=agent, environment=env, step=step)
    simulator.run(visualize=visualize, n_steps_vis=n_steps_vis, log=log,
                  each_step_propagation=each_step_propagation, print_out=print_out)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
