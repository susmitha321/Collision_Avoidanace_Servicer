# Generate random collision situation.
import argparse
import sys

import numpy as np
import pykep as pk
import time

from .generator_utils import SpaceObject2srt, rotate_velocity
from ..api import SpaceObject


class Generator:
    """Generates random collision situation.

    TODO:
        better distributions parameters
        add user protected object?
        add user collision times?
    """

    def __init__(self, start_time, end_time, servicer_size=1):
        # TODO - random start/end time with duration?
        # TODO - random duration?
        np.random.seed(int(time.time()))
        self.start_time = pk.epoch(start_time, "mjd2000")
        self.end_time = pk.epoch(end_time, "mjd2000")

        self.protected = None
        self.servicer = None
        self.debris = []
        self.servicer_size = servicer_size
        self.collision_epochs = []

    def add_protected(self):
        """Add protected object."""

        # gravity parameter of the object (m^2/s^3)
        # TODO - add distribution?
        mu_self = 0.1

        # objects radii (meters)
        """
        http://www.businessinsider.com/size-of-most-famous-satellites-2015-10
        """
        radius = np.random.uniform(0.3, 55)

        # mimimual radius that is safe during a fly-by of the object (meters)
        # TODO - differs from radius?
        safe_radius = radius

        # protected fuel
        # TODO - add distribution?
        fuel = 15

        # six osculating keplerian elements (a,e,i,W,w,M) at the reference epoch
        # a (semi-major axis): meters
        # https://upload.wikimedia.org/wikipedia/commons/b/b4/Comparison_satellite_navigation_orbits.svg
        a = np.random.uniform(7e6, 8e6)
        # e (eccentricity): in interval [0, 1)
        e = np.random.uniform(0, 0.003)
        # i (inclination): radians
        # TODO - fix if i > pi? i > 2*pi?
        i = np.random.uniform(0, 2 * np.pi)
        # W (Longitude of the ascending node): radians
        W = np.random.uniform(0, 2 * np.pi)
        # w (Argument of periapsis): radians
        w = np.random.uniform(0, 2 * np.pi)
        # M (mean anomaly): radians
        M = np.random.uniform(0, 2 * np.pi)
        # Keplerian elements
        elements = [a, e, i, W, w, M]
        self.elements_pro = elements
        # protected object parameters
        params = {
            "epoch": self.start_time,
            "elements": elements,
            "mu_central_body": pk.MU_EARTH,
            "mu_self": mu_self,
            "radius": radius,
            "safe_radius": safe_radius,
            "fuel": fuel
        }

        self.protected = SpaceObject("PROTECTED", "osc", params)
        print(self.protected.satellite.osculating_elements(params["epoch"]))
        #print(dir(self.protected.satellite.osculating_elements))
    
    def add_servicer(self):
        """Add Servicer object."""

        # gravity parameter of the object (m^2/s^3)
        # TODO - add distribution?
        mu_self = 0.1

        # objects radii (meters) distance from cube center to one of its corners
        """
        http://www.businessinsider.com/size-of-most-famous-satellites-2015-10
        """
        side = self.servicer_size
        radius = np.sqrt((side/2)**2 + (side/np.sqrt(2))**2)
        
        # mimimual radius that is safe during a fly-by of the object (meters)
        # TODO - differs from radius?
        safe_radius = radius

        # protected fuel
        # TODO - add distribution?
        fuel = 15

        
        # Retrieve the Keplerian elements from the protected object
        a, e, i, W, w, M_protected = self.elements_pro
        #self.protected.satellite.osculating_elements(self.start_time)

        # Introduce a random phase difference for the servicer by adjusting the mean anomaly
        phase_difference = 7*np.pi/180
        #np.random.uniform(0, 2 * np.pi)
        M_servicer = (M_protected + phase_difference) % (2 * np.pi)  # Ensure it's within [0, 2*pi]
        
        # check if the protected and servicer are close to each other, then run this loop until they are not.
        
        while np.isclose(M_servicer, M_protected, atol=1e-8):  # Using a small tolerance to check for equality
            phase_difference = np.random.uniform(0, 2 * np.pi)
            M_servicer = (M_protected + phase_difference) % (2 * np.pi)
       
        # Keplerian elements for the servicer
        elements = [a, e, i, W, w, M_servicer]

        # servicer object parameters
        params = {
                     "epoch": self.start_time,
                     "elements": elements,
                     "mu_central_body": pk.MU_EARTH,
                     "mu_self": mu_self,
                     "radius": radius,
                     "safe_radius": safe_radius,
                     "fuel": fuel
                 }

        self.servicer = SpaceObject("SERVICER", "osc", params)
        print(self.servicer.satellite.osculating_elements(params["epoch"]))
    def add_debris(self, pos_sigma=0, vel_ratio_sigma=0.05,
                   i_threshold=0.5):
        """Add debris object.

        Args:
            pos_sigma (float): standard deviation of debris position
                from protected one (meters).
            vel_ratio_sigma (float): standard deviation of debris and protected
                velocities ratio (m/s).
            i_threshold (float): minimum angle between debris and protected
                at collision time (radians) (<=pi/4).

        Raises:
            Exception: if the protected object was not added.

        TODO:
            add ValueErrors for args.

        """
        if not self.protected:
            raise Exception("no protected object")
            
        ### Make this modification 
        #Generate random collision only from start time/2 to end time maybe?
        # 
        
        # TODO - indent?
        
        collision_time = np.random.uniform(
            self.start_time.mjd2000, self.end_time.mjd2000)
        #+ self.end_time.mjd2000)/2
        collision_time = pk.epoch(collision_time, "mjd2000")
        self.collision_epochs.append(collision_time)

        # position (x, y, z) and velocity (Vx, Vy, Vz) of protected object
        pos_prot, vel_prot = self.protected.position(collision_time)

        # position and velocity of debris at collision time
        # TODO -  truncated normal?
        pos = np.random.normal(pos_prot, pos_sigma)
        rotate_angle = np.random.choice([
            np.random.uniform(i_threshold, np.pi - i_threshold),
            np.random.uniform(np.pi + i_threshold, 2 * np.pi - i_threshold)
        ])
        vel = rotate_velocity(vel_prot, pos, rotate_angle)
        vel = vel * np.random.normal(1, vel_ratio_sigma)

        # gravity parameter of the object (m^2/s^3)
        # TODO - add distribution?
        mu_self = 0.1

        # objects radii (meters)
        """
        https://www.nasa.gov/mission_pages/station/news/orbital_debris.html
        https://m.esa.int/Our_Activities/Operations/Space_Debris/Space_debris_by_the_numbers
        """
        radius = np.random.uniform(0.05, 1)

        # mimimual radius that is safe during a fly-by of the object (meters)
        # TODO - differs from radius?
        safe_radius = radius

        name = "DEBRIS" + str(len(self.debris))

        # protected object parameters
        params = {
            "epoch": collision_time,
            "pos": pos,
            "vel": vel,
            "mu_central_body": pk.MU_EARTH,
            "mu_self": mu_self,
            "radius": radius,
            "safe_radius": safe_radius,
            "fuel": 0
        }

        self.debris.append(SpaceObject(name, "eph", params))

    def save_env(self, save_path, time_before_start_time=0):
        with open(save_path, 'w') as f:
            f.write(f'{self.start_time.mjd2000 - time_before_start_time}, {self.end_time.mjd2000}\n')
            f.write('osc\n')
            f.write(SpaceObject2srt(self.protected, self.start_time))
            if self.servicer:  # Save servicer details if it exists
                #f.write('osc\n')
                f.write(SpaceObject2srt(self.servicer, self.start_time))
            for debr, epoch in zip(self.debris, self.collision_epochs):
                f.write(SpaceObject2srt(debr, epoch))

    def env(self):
        pass

    def info(self):
        pass
