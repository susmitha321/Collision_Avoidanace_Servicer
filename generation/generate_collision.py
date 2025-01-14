# Generate random collision environment.

import argparse
import sys
import os

from space_navigator.generator import Generator


def main(args):
    parser = argparse.ArgumentParser()

    # generator initialization
    parser.add_argument("-n_d", "--n_debris", type=int,
                        default=2, required=False)
    parser.add_argument("-start", "--start_time", type=float,
                        default=6600, required=False)
    parser.add_argument("-end", "--end_time", type=float,
                        default=6600.1, required=False)
    parser.add_argument("-before", "--time_before_start_time", type=float,
                        default=0, required=False)
    parser.add_argument("-s_s","--servicer_size",type=int,
                        default=1, required=False)
    

    # debris parameters
    parser.add_argument("-p_s", "--pos_sigma", type=float,
                        default=0, required=False)
    parser.add_argument("-v_r_s", "--vel_ratio_sigma", type=float,
                        default=0.05, required=False)
    parser.add_argument("-i_t", "--i_threshold", type=float,
                        default=0.5, required=False)

    # other parameters
    parser.add_argument("-save_path", "--environment_save_path", type=str,
                        default="data/environments/generated_collision.env", required=False)

    # args parsing
    args = parser.parse_args(args)

    n_debris, start_time, end_time, servicer_size  = args.n_debris, args.start_time, args.end_time, args.servicer_size
    time_before_start_time = args.time_before_start_time

    pos_sigma, vel_ratio_sigma = args.pos_sigma, args.vel_ratio_sigma
    i_threshold = args.i_threshold

    save_path = args.environment_save_path
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # generation
    generator = Generator(start_time, end_time, servicer_size)

    generator.add_protected()
    #print(self.protected.satellite.osculating_parameters(start_time))
    generator.add_servicer()
    #print(self.servicer.satellite.osculating_parameters(self.start_time))
#   generator.add_dummy()
    #print(self.dummy.satellite.osculating_parameters(self.start_time))
    
    for _ in range(n_debris):
        generator.add_debris(pos_sigma, vel_ratio_sigma, i_threshold)

    generator.save_env(save_path, time_before_start_time)

    return

if __name__ == "__main__":
    main(sys.argv[1:])
